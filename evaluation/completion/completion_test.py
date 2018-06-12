# -*- coding: utf-8 -*-
# File   : test.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 25/01/2018
#
# This file is part of Semantic-Graph-PyTorch.

import json
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
from jaclearn.embedding.word_embedding import load as load_word_embedding
from jactorch.cuda.copy import async_copy_to
from jactorch.io import load_weights
from jactorch.utils.meta import as_numpy, mark_volatile
from evaluation.completion.cli import ensure_path, format_meters, dump_metainfo
from evaluation.completion.dataset import CompletionDataset, make_dataloader
from evaluation.completion.model import CompletionModel

from vocab import Vocabulary

logger = get_logger(__file__)

parser = JacArgumentParser(description='Semantic graph testing')
parser.add_argument('--load', required=True, type='checked_dir', metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--mode', default='all', choices=['all', 'noun', 'prep'], metavar='M')

parser.add_argument('--use-gpu', default=True, type='bool', metavar='B', help='use GPU or not')

parser.add_argument('--vse', required=True, type='checked_file', metavar='FILE', help='vse file')
parser.add_argument('--glove-only', action='store_true')
parser.add_argument('--data-dir', required=True, type='checked_dir', help='data directory')
parser.add_argument('--dev-img', default='dev_ims.npy', metavar='FILE', help='dev data json file')
parser.add_argument('--dev-cap', default='dev_caps_replace.json', metavar='FILE', help='dev data json file')
parser.add_argument('--test-img', default='test_ims.npy', metavar='FILE', help='testing data json file')
parser.add_argument('--test-cap', default='test_caps_replace.json', metavar='FILE', help='testing data json file')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input testing data')

args = parser.parse_args()


if args.use_gpu:
    nr_devs = cuda.device_count()
    assert nr_devs > 0, 'No GPU device available'


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
    logger.critical('Loading the word embedding.')
    vocab, word_embeddings = load_word_embedding(args.vse)

    logger.critical('Building up the model.')
    model = CompletionModel(word_embeddings)
    if args.use_gpu:
        model.cuda()
    # Disable the cudnn benchmark.
    model.eval()
    cudnn.benchmark = False

    logger.critical('Loading the dataset.')

    dev_dataset = CompletionDataset(vocab, pjoin(args.data_dir, args.dev_img), pjoin(args.data_dir, args.dev_cap), mode=args.mode)
    test_dataset = CompletionDataset(vocab, pjoin(args.data_dir, args.test_img), pjoin(args.data_dir, args.test_cap), mode=args.mode)

    logger.critical('Building up the data loader.')
    dev_dataloader = make_dataloader(dev_dataset, num_workers=args.data_workers, batch_size=64, shuffle=False, drop_last=False, pin_memory=True)
    test_dataloader = make_dataloader(test_dataset, num_workers=args.data_workers, batch_size=64, shuffle=False, drop_last=False, pin_memory=True)

    for epoch_id in range(1, 11):
        load_weights(model, pjoin(args.load, 'epoch_{}.pth'.format(epoch_id)))

        for loader in [dev_dataloader, test_dataloader]:
            meters = GroupMeters()

            end = time.time()
            with tqdm_pbar(total=len(loader), leave=False) as pbar:
                for i, data in enumerate(loader):
                    feed_dict = data
                    feed_dict = mark_volatile(feed_dict)

                    if args.use_gpu:
                        feed_dict = async_copy_to(feed_dict, 0)

                    data_time = time.time() - end; end = time.time()

                    output_dict = model(feed_dict)
                    output_dict = as_numpy(output_dict)

                    gpu_time = time.time() - end;  end = time.time()

                    meters.update({k: float(v) for k, v in output_dict.items() if k.startswith('top')}, n=len(feed_dict['image']))
                    meters.update({'time/data': data_time, 'time/gpu': gpu_time})

                    pbar.set_description(format_meters('sentid={}'.format(i), meters.val, '{}={:.4f}', ', '))
                    pbar.update()

                    end = time.time()

            print(epoch_id, sorted(meters.avg.items()))


if __name__ == '__main__':
    main()
