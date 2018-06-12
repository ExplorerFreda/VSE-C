import six
import collections
from os.path import join as pjoin
import warnings
warnings.filterwarnings("ignore")  # skimage warnings

from PIL import Image
from skimage.transform import resize as imresize

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import jacinle.io as io
from jacinle.cli.argument import JacArgumentParser
from jacinle.utils.enum import JacEnum
from jactorch.utils.meta import as_variable, as_cuda
from jactorch.graph.variable import var_with

from vocab import Vocabulary

Record = collections.namedtuple('Record', [
    'raw_image', 'raw_caption', 'raw_caption_ext',
    'image', 'image_embedding', 'image_embedding_precomp',
    'captions', 'caption_embedding', 'caption_ext_embedding'
])

parser = JacArgumentParser()
parser.add_argument('--load', required=True)
parser.add_argument('--encoder', required=True)
parser.add_argument('--load-encoder', required=True, type='checked_file')
parser.add_argument('--images', required=True, type='checked_dir')
parser.add_argument('--image-list', required=True, type='checked_file')
parser.add_argument('--image-embeddings', required=True, type='checked_file')
parser.add_argument('--captions', required=True, type='checked_file')
parser.add_argument('--captions-ext', required=True, type='checked_dir')

args = parser.parse_args()
args.grad_power = 0.5


def main():
    encoder = ImageEncoder(args.encoder, args.load_encoder)
    dataset = Dataset(args)
    extractor = FeatureExtractor(args.load, encoder, dataset)

    def e(ind):
        a = extractor(ind)
        pic = plot_saliency(a.raw_image, a.image, a.image_embedding, a.caption_embedding)
        pic.save('/tmp/origin{}.png'.format(ind))
        print('Image saliency saved:', '/tmp/origin{}.png'.format(ind))
        print_txt_saliency(a.captions, 0, a.raw_caption, a.image_embedding, a.caption_embedding)
        return a

    from IPython import embed; embed()


def normalize_grad(grad, stat=False):
    grad = np.abs(grad)
    if stat:
        print('Grad min={}, max={}'.format(grad.min(), grad.max()))
    grad -= grad.min()
    grad /= grad.max()
    return grad.astype('float32')


def print_txt_saliency(txt, ind, content, img_embedding_var, caption_embedding_var, backward=False):
    if backward:
        dis = (caption_embedding_var.squeeze() * img_embedding_var.squeeze()).sum()
        dis.backward(retain_graph=True)

    content = content.split()
    assert txt.grad.size(1) == 2 + len(content), (txt.grad.size(1), len(content))
    grad_txt = txt.grad[ind,1:1+len(content)].data.cpu().squeeze().abs().numpy()
    grad_txt = grad_txt.mean(-1)
    grad_txt /= grad_txt.sum()
    print('Text saliency:', ' '.join(['{}({:.3f})'.format(c, float(g)) for c, g in zip(content, grad_txt)]))


def plot_saliency(raw_img, image_var, img_embedding_var, caption_var):
    dis = (caption_var.squeeze() * img_embedding_var.squeeze()).sum()
    dis.backward(retain_graph=True)

    grad = image_var.grad.data.cpu().squeeze().numpy().transpose((1, 2, 0))
    grad = normalize_grad(grad, stat=True)
    grad = imresize((grad * 255).astype('uint8'), (raw_img.height, raw_img.width)) / 255
    grad = normalize_grad(grad.mean(axis=-1, keepdims=True).repeat(3, axis=-1))
    grad = np.float_power(grad, args.grad_power)

    np_img = np.array(raw_img)
    masked_img = np_img * grad
    final = np.hstack([np_img, masked_img.astype('uint8'), (grad * 255).astype('uint8')])
    return Image.fromarray(final.astype('uint8'))


def plot(record, ind, pic_ind=None):
    p = plot_saliency(record.raw_image, record.image, record.image_embedding, record.caption_ext_embedding[ind])
    if pic_ind is not None:
        name = '/tmp/{}_{}.png'.format(pic_ind, ind)
    else:
        name = '/tmp/{}.png'.format(ind)
    p.save(name)
    print('Image saliency saved:', name)
    print_txt_saliency(record.captions, ind + 1, record.raw_caption_ext[ind], record.image_embedding, record.caption_ext_embedding[ind])


class ImageEncoderType(JacEnum):
    RESNET152 = 'resnet152'


class Identity(nn.Module):
    def forward(self, x):
        return x


def ImageEncoder(encoder, load):
    encoder = ImageEncoderType.from_string(encoder)
    if encoder is ImageEncoderType.RESNET152:
        encoder = models.resnet152()
        encoder.load_state_dict(torch.load(load))
        encoder.fc = Identity()
        encoder.cuda()
        encoder.eval()
        return encoder


class Dataset(object):
    def __init__(self, args):
        self.images = args.images
        self.image_list = args.image_list
        self.image_embeddings = args.image_embeddings
        self.captions = args.captions
        self.captions_ext = args.captions_ext

        self.load_files()
        self.build_image_transforms()

    def get_image(self, ind):
        img_file = self.image_list[ind].strip()
        if 'train2014' in img_file:
            img_file = pjoin('train2014', img_file)
        elif 'val2014' in img_file:
            img_file = pjoin('val2014', img_file)
        img = Image.open(pjoin(self.images, img_file)).convert('RGB')
        return img

    def get_image_embedding(self, ind):
        return self.image_embeddings[ind]

    def get_caption(self, ind):
        return self.captions[ind].strip()

    def get_caption_ext(self, ind):
        with open('{}/{}.txt'.format(self.captions_ext, str(ind))) as f:
            return [line.strip() for line in f.readlines()]

    def load_files(self):
        print('Loading captions and precomputed image embeddings')
        self.image_embeddings = io.load(self.image_embeddings)
        self.captions = list(open(self.captions))
        image_list_dup = list()
        image_list = list(open(self.image_list))
        assert len(image_list) * 5 == len(self.captions)
        for img in image_list:
            for i in range(5):
                image_list_dup.append(img)
        self.image_list = image_list_dup

    def build_image_transforms(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, ind):
        img = self.get_image(ind)
        img_embedding = self.get_image_embedding(ind)
        cap = self.get_caption(ind)
        cap_ext = self.get_caption_ext(ind)

        return img, self.image_transform(img), img_embedding, cap, cap_ext


class FeatureExtractor(object):
    def __init__(self, checkpoint, image_encoder, dataset):
        self.load_checkpoint(checkpoint)
        self.image_encoder = image_encoder
        self.dataset = dataset

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        opt = checkpoint['opt']
        opt.use_external_captions = False
        vocab = Vocab.from_pickle(pjoin(opt.vocab_path, '%s_vocab.pkl' % opt.data_name))
        opt.vocab_size = len(vocab)

        from model import VSE
        self.model = VSE(opt)
        self.model.load_state_dict(checkpoint['model'])
        self.projector = vocab

        self.model.img_enc.eval()
        self.model.txt_enc.eval()
        for p in self.model.img_enc.parameters():
            p.requires_grad = False
        for p in self.model.txt_enc.parameters():
            p.requires_grad = False

    def __call__(self, ind):
        raw_img, img, img_embedding, cap, cap_ext = self.dataset[ind]
        img_embedding_precomp = self.model.img_enc(as_cuda(as_variable(img_embedding).unsqueeze(0)))

        img = as_variable(img)
        img.requires_grad = True
        img_embedding_a = img_embedding = self.image_encoder(as_cuda(img.unsqueeze(0)))
        img_embedding = self.model.img_enc(img_embedding)

        txt = [cap]
        txt.extend(cap_ext)
        txt_embeddings, txt_var = self.enc_txt(txt)

        return Record(
                raw_img, cap, cap_ext,
                img, img_embedding, img_embedding_precomp,
                txt_var, txt_embeddings[0], txt_embeddings[1:]
        )

    def enc_txt(self, caps):
        sents, lengths, _, inv = _prepare_batch(caps, self.projector)
        inv = var_with(as_variable(inv), sents)
        out, x = self.model.txt_enc.forward(sents, lengths, True)
        return out[inv], x


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

    def project(self, sentence):
        sentence = sentence.strip().lower().split()
        sentence = ['<start>'] + sentence + ['<end>']
        if self.sent_trunc_length is not None:
            if len(sentence) > self.sent_trunc_length:
                sentence = sentence[:self.sent_trunc_length]
        return list(map(lambda word: self.word2idx.get(word, 3), sentence))

    def __len__(self):
        return len(self.idx2word)
    
    def __call__(self, sent):
        return self.project(sent)


def _prepare_batch(sents, projector):
    if isinstance(sents, six.string_types):
        sents = [sents]

    sents = [np.array(projector(s)) for s in sents]
    lengths = [len(s) for s in sents]
    sents = _pad_sequences(sents, 0, max(lengths))

    idx = np.array(sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True))
    inv = np.array(sorted(range(len(lengths)), key=lambda x: idx[x]))
    sents = sents[idx]
    lengths = np.array(lengths)[idx].tolist()

    sents = as_variable(sents)
    if torch.cuda.is_available():
        sents = sents.cuda()
    return sents, lengths, idx, inv


def _pad_sequences(sequences, dim, length):
    seq_shape = list(sequences[0].shape)
    seq_shape[dim] = length
    output = np.zeros((len(sequences), ) + tuple(seq_shape), dtype=sequences[0].dtype)
    output = output.reshape((len(sequences), -1) + tuple(seq_shape[dim:]))
    for i, seq in enumerate(sequences):
        output[i, :, :seq.shape[dim], ...] = seq
    return output.reshape((len(sequences), ) + tuple(seq_shape))

if __name__ == '__main__':
    main()

