# -*- coding: utf-8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
#
# This file is part of Semantic-Graph-PyTorch.

import os.path as osp
from os.path import join as pjoin
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

import jacinle.io as io
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.meter import GroupMeters
from jacinle.utils.tqdm import tqdm_pbar
from jactorch.cuda.copy import async_copy_to
from jactorch.optim import AccumGrad, AdamW
from jactorch.train import TrainerEnv
from jactorch.train.tb import TBLogger, TBGroupMeters
from evaluation.completion.cli import ensure_path, format_meters, dump_metainfo
from evaluation.completion.dataset import CompletionDataset, make_dataloader
from evaluation.completion.model import CompletionModel

from vocab import Vocabulary

logger = get_logger(__file__)

parser = JacArgumentParser(description='Completion training')
parser.add_argument('--desc', required=True, help='description name')
parser.add_argument('--use-gpu', default=True, type='bool', metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', default=True, type='bool', metavar='B', help='use tensorboard or not')

parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='batch size')
parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='initial learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float, metavar='N', help='weight decay')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset)')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')

parser.add_argument('--resume', type='checked_file', metavar='FILE', help='path to latest checkpoint (default: none)')
parser.add_argument('--load', type='checked_file', metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N', help='model save interval (epochs)')

parser.add_argument('--vse', required=True, type='checked_file', metavar='FILE', help='vse file')
parser.add_argument('--glove-only', action='store_true')
parser.add_argument('--data-dir', required=True, type='checked_dir', help='data directory')
parser.add_argument('--train-img', default='train_ims.npy', metavar='FILE', help='training data json file')
parser.add_argument('--train-cap', default='train_caps_replace.json', metavar='FILE', help='training data json file')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

parser.add_argument('--embed', default=False, action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', default=False, action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# make the model name
args.desc_name = args.desc
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
args.dump_dir = ensure_path(osp.join('mjy_runs', args.desc_name))
args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')

if args.use_tb:
    args.tb_dir_root = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
    args.tb_dir = ensure_path(osp.join(args.tb_dir_root, args.run_name))


if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)


class Vocab(object):
    def __init__(self, idx2word=None, options=None, sync=None):
        assert options is None
        if sync is not None:
            self.idx2word = sync.idx2word
            self.word2idx = sync.word2idx
        else:
            self.idx2word = idx2word
            self.word2idx = dict([(w, i) for i, w in enumerate(self.idx2word)])
        self.sent_trunc_length = None

    @classmethod
    def from_pickle(cls, path):
        vocab = io.load(path)
        return cls(sync=vocab)

    def project(self, sentence, is_sent=True):
        sentence = sentence.strip().lower().split()
        if is_sent:
            sentence = ['<start>'] + sentence + ['<end>']
        if self.sent_trunc_length is not None:
            if len(sentence) > self.sent_trunc_length:
                sentence = sentence[:self.sent_trunc_length]
        return list(map(lambda word: self.word2idx.get(word, 3), sentence))

    def __len__(self):
        return len(self.idx2word)
    
    def __call__(self, sent, is_sent=True):
        return self.project(sent, is_sent=is_sent)


def load_word_embedding(vse):
    checkpoint = torch.load(vse)
    opt = checkpoint['opt']
    vocab = Vocab.from_pickle(pjoin(opt.vocab_path, '%s_vocab.pkl' % opt.data_name))

    if not args.glove_only:
        embed_weights = checkpoint['model'][1]['embed.weight'].cpu().numpy()
        _, glove_weights = io.load('data/snli/glove.pkl')
        embed_weights = np.concatenate((glove_weights, embed_weights), axis=1)
    else:
        _, embed_weights = io.load('data/snli/glove.pkl')
    embedding = nn.Embedding(embed_weights.shape[0], embed_weights.shape[1], padding_idx=0)
    embedding.weight.data.copy_(torch.from_numpy(embed_weights))
    return vocab, embedding


def main():
    logger.critical('Writing metainfo to file: {}.'.format(args.meta_file))

    with open(args.meta_file, 'w') as f:
        f.write(dump_metainfo(args=args.__dict__))

    logger.critical('Loading the word embedding.')
    vocab, word_embeddings = load_word_embedding(args.vse)

    logger.critical('Building up the model.')
    model = CompletionModel(word_embeddings)
    if args.use_gpu:
        model.cuda()
        assert not args.gpu_parallel
    # Disable the cudnn benchmark.
    cudnn.benchmark = False

    logger.critical('Loading the dataset.')

    train_dataset = CompletionDataset(vocab, pjoin(args.data_dir, args.train_img), pjoin(args.data_dir, args.train_cap))

    logger.critical('Building up the data loader.')
    train_loader = make_dataloader(train_dataset, num_workers=args.data_workers, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # Default optimizer.
    optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), args.lr, weight_decay=args.weight_decay)
    if args.acc_grad > 1:
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))
    trainer = TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: {}.'.format(args.load))

    if args.use_tb:
        logger.critical('Writing tensorboard logs to: {}.'.format(args.tb_dir))
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
    else:
        meters = GroupMeters()

    # Switch to train mode.
    model.train()

    if args.embed:
        from IPython import embed; embed()
        return

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        train_epoch(epoch, train_loader, trainer, meters)

        logger.critical(format_meters('Epoch = {}'.format(epoch), meters.avg, '\t{} = {:.4f}', '\n'))

        if epoch % args.save_interval == 0:
            fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))


def train_epoch(epoch, train_loader, trainer, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_loader)

    meters.reset()
    end = time.time()

    trainer.trigger_event('epoch:before', trainer, epoch)

    train_iter = iter(train_loader)
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)

            if not args.gpu_parallel and args.use_gpu:
                feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict)
            step_time = time.time() - end;  end = time.time()

            meters.update(loss=loss)
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(format_meters('iter={}/{}'.format(epoch, i),
                {k: v for k, v in meters.val.items() if k.startswith('loss') or k.startswith('time')},
                '{}={:.4f}', ', '))
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


if __name__ == '__main__':
    main()
