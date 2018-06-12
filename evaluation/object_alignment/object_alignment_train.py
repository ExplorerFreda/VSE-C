import torch
from torch import nn
from torch.autograd import Variable
from VSE_C.model import VSE
from VSE_C.vocab import *
from VSE_C.data import *


class ObjectAlignmentNet(nn.Module):
    def __init__(self, w2v, im_dim):
        super(ObjectAlignmentNet, self).__init__()
        self.embed = nn.Embedding(w2v.shape[0], w2v.shape[1])
        self.embed.weight.data = torch.FloatTensor(w2v)
        self.im_dim = im_dim
        self.word_dim = w2v.shape[1]
        self.alignment = nn.Linear(self.im_dim * self.word_dim, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, ims, words):
        words = self.embed(words).view(words.size(0), -1, 1)
        ims = ims.view(ims.size(0), 1, -1)
        features = torch.bmm(words, ims).view(ims.size(0), -1)
        return self.alignment(self.dropout(features))


def train(data_path, split, vocab, batch_size=128):
    model.train()
    ims = np.load(pjoin(data_path, split + '_objs.npy'))
    p_objs = [line.strip().split() for line in open(pjoin(data_path, split + '_objs.txt')).readlines()]
    n_objs = [line.strip().split() for line in open(pjoin(data_path, split + '_nobjs.txt')).readlines()]
    ids = list(filter(lambda x: (len(p_objs[x]) > 0) and (len(n_objs[x]) > 0), [i for i in range(ims.shape[0])]))
    random.shuffle(ids)
    total_acc = 0
    total_loss = 0
    cnt = 0
    for bid in range(0, len(ids), batch_size):
        b_im = Variable(torch.FloatTensor(np.stack(
            [ims[idx] for idx in ids[bid: min(len(ids), batch_size + bid)]])))
        b_p_objs = Variable(torch.LongTensor(
            [vocab.word2idx.get(p_objs[idx][random.randint(0, len(p_objs[idx]) - 1)], 3)
             for idx in ids[bid: min(len(ids), batch_size + bid)]]))
        b_n_objs = Variable(torch.LongTensor(
            [vocab.word2idx.get(n_objs[idx][random.randint(0, len(n_objs[idx]) - 1)], 3)
             for idx in ids[bid: min(len(ids), batch_size + bid)]]))
        labels = Variable(torch.LongTensor([1 for _ in range(b_im.size(0))] + [0 for _ in range(b_im.size(0))]))
        if torch.cuda.is_available():
            b_im = b_im.cuda()
            b_p_objs = b_p_objs.cuda()
            b_n_objs = b_n_objs.cuda()
            labels = labels.cuda()
        output_p = model(b_im, b_p_objs)
        output_n = model(b_im, b_n_objs)
        output = torch.cat((output_p, output_n), dim=0)
        loss = criterion(output, labels)

        total_acc += torch.sum((output.max(1)[1] == labels).long()).data[0]
        total_loss += loss.data[0] * labels.size(0)
        cnt += labels.size(0)
        if (bid // batch_size) % 100 == 0:
            print((bid // batch_size) // 100, 'loss:', total_loss / cnt, '| acc:', total_acc / cnt * 100)
            fout.write('loss: {:5f} | acc: {:3f}\n'.format(total_loss / cnt, total_acc / cnt * 100))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid(data_path, split, vocab):
    model.eval()
    ims = np.load(pjoin(data_path, split + '_objs.npy'))
    p_objs = [line.strip().split() for line in open(pjoin(data_path, split + '_objs.txt')).readlines()]
    n_objs = [line.strip().split() for line in open(pjoin(data_path, split + '_nobjs.txt')).readlines()]
    ids = list(filter(lambda x: (len(p_objs[x]) > 0) and (len(n_objs[x]) > 0), [i for i in range(ims.shape[0])]))
    full_map = 0
    for idx in ids:
        select_p_ids = list()
        select_n_ids = list()
        select_p_words = list()
        select_n_words = list()
        for _ in range(len(p_objs[idx])):
            select_p_ids.append(idx)
            select_p_words.append(p_objs[idx][_])
        for _ in range(len(n_objs[idx])):
            select_n_ids.append(idx)
            select_n_words.append(n_objs[idx][_])
        b_im_p = Variable(torch.FloatTensor(np.stack([ims[idx] for idx in select_p_ids])), volatile=True)
        b_im_n = Variable(torch.FloatTensor(np.stack([ims[idx] for idx in select_n_ids])), volatile=True)
        b_objs_p = Variable(torch.LongTensor([vocab.word2idx.get(word, 3) for word in select_p_words]), volatile=True)
        b_objs_n = Variable(torch.LongTensor([vocab.word2idx.get(word, 3) for word in select_n_words]), volatile=True)
        if torch.cuda.is_available():
            b_im_p = b_im_p.cuda()
            b_im_n = b_im_n.cuda()
            b_objs_p = b_objs_p.cuda()
            b_objs_n = b_objs_n.cuda()
        output_p = model(b_im_p, b_objs_p)
        output_n = model(b_im_n, b_objs_n)
        output = torch.cat((output_p, output_n), dim=0)

        # compute MAP
        correct_cnt = 0
        _, poses = output[:, 1].sort(descending=True)
        mean_ap = 0
        for i, pos in enumerate(poses.cpu().data.numpy().tolist()):
            if pos < b_objs_p.size(0):
                correct_cnt += 1
                ap = correct_cnt / float(i + 1)
                mean_ap += ap
        if correct_cnt == 0:
            mean_ap = 1
        else:
            mean_ap /= correct_cnt
        full_map += mean_ap
    full_map /= len(ids)
    print(split)
    print('MAP:', full_map * 100)
    fout.write(split + '\n')
    fout.write('MAP: {:5f}% \n'.format(full_map * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='vse')
    parser.add_argument('--model_path', type=str, default='group7_word_embedding_study/relation_ex')
    parser.add_argument('--data', type=str, default='data/resnet152_precomp')
    parser.add_argument('--no-concat', dest='concat', action='store_false')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--decay', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    if args.type == 'vse':
        checkpoint = torch.load(pjoin(args.model_path, 'model_best.pth.tar'))
        opt = checkpoint['opt']
        opt.use_external_captions = False
        # load vocabulary used by the model
        with open(pjoin(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
            vocab = pickle.load(f)
        opt.vocab_size = len(vocab)
        # construct model
        loaded_model = VSE(opt)
        loaded_model.load_state_dict(checkpoint['model'])
        if opt.word_dim > 300:
            encoder_weight = loaded_model.txt_enc.embed.weight.data.cpu().numpy()
            if args.concat:
                _, encoder_weight_ = pickle.load(open('data/snli/glove.pkl', 'rb'))
                encoder_weight = np.concatenate((encoder_weight, encoder_weight_), axis=1)
        else:
            _, encoder_weight = pickle.load(open('data/snli/glove.pkl', 'rb'))
    else:
        assert args.type == 'glove'
        _, encoder_weight = pickle.load(open('data/snli/glove.pkl', 'rb'))
        with open('vocab/resnet152_precomp_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

    if args.data.find('resnet') != -1:
        im_dim = 2048
    else:
        im_dim = 4096
    model = ObjectAlignmentNet(encoder_weight, im_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = getattr(torch.optim, args.optimizer)(model.alignment.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    fout = open('object_alignment_eval/' + args.model_path[args.model_path.rfind('/') + 1:] + '.log', 'w')
    for epoch in range(args.epochs):
        if epoch % args.decay == args.decay - 1:
            args.lr = args.lr / 10.0
            optimizer = getattr(torch.optim, args.optimizer)(model.alignment.parameters(), lr=args.lr)
        print('Epoch:', epoch)
        fout.write('Epoch: {:d}\n'.format(epoch))
        train(args.data, 'train', vocab, args.batch_size)
        valid(args.data, 'dev', vocab)
        valid(args.data, 'test', vocab)
    fout.close()
