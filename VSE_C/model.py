import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn import functional
import numpy as np
from collections import OrderedDict
from IPython import embed
import pickle
import random
from torch.autograd import Variable
from os.path import join as pjoin


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt().view(X.size(0), -1)
    X = torch.div(X, norm.expand_as(X))
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print(("=> using pre-trained model '{}'".format(arch)))
            model = models.__dict__[arch](pretrained=True)
        else:
            print(("=> creating model '{}'".format(arch)))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in list(state_dict.items()):
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)

    def __call__(self, images):
        return self.forward(images)


class Combiner(nn.Module):
    def __init__(self, pooling, hidden_size=1024, attention_size=128):
        super(Combiner, self).__init__()
        self.method = pooling
        if self.method == 'attn':
            self.ws1 = nn.Linear(hidden_size, attention_size, bias=False)
            self.ws2 = nn.Linear(attention_size, 1, bias=False)
            self.tanh = nn.Tanh()

    def forward(self, encoding, lengths):
        lengths = Variable(torch.LongTensor(lengths))
        if torch.cuda.is_available():
            lengths = lengths.cuda()
        if self.method == 'mean':
            encoding_pad = nn.utils.rnn.pack_padded_sequence(encoding, lengths.data.tolist(), batch_first=True)
            encoding = nn.utils.rnn.pad_packed_sequence(encoding_pad, batch_first=True, padding_value=0)[0]
            lengths = lengths.float().view(-1, 1)
            return encoding.sum(1) / lengths, None
        elif self.method == 'max':
            return encoding.max(1)  # [bsz, in_dim], [bsz, in_dim] (position)
        elif self.method == 'attn':
            size = encoding.size()  # [bsz, len, in_dim]
            x_flat = encoding.contiguous().view(-1, size[2])  # [bsz*len, in_dim]
            hbar = self.tanh(self.ws1(x_flat))  # [bsz*len, attn_hid]
            alphas = self.ws2(hbar).view(size[0], size[1])  # [bsz, len]
            alphas = nn.utils.rnn.pack_padded_sequence(alphas, lengths.data.tolist(), batch_first=True)
            alphas = nn.utils.rnn.pad_packed_sequence(alphas, batch_first=True, padding_value=-1e8)[0]
            alphas = functional.softmax(alphas, dim=1)  # [bsz, len]
            alphas = alphas.view(size[0], 1, size[1])  # [bsz, 1, len]
            return torch.bmm(alphas, encoding).squeeze(1), alphas  # [bsz, in_dim], [bsz, len]
        elif self.method == 'last':
            return torch.cat([encoding[i][lengths[i] - 1] for i in range(encoding.size(0))], dim=0), None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderTextGRU(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, pooling='last',
                 use_abs=False, bid=False, glove_path='data/glove.pkl'):
        super(EncoderTextGRU, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.combiner = Combiner(pooling, embed_size)

        # word embedding
        self.word_dim = word_dim
        if word_dim > 300:
            self.embed = nn.Embedding(vocab_size, word_dim-300)
        _, embed_weight = pickle.load(open(glove_path, 'rb'))
        self.glove = Variable(torch.cuda.FloatTensor(embed_weight), requires_grad=False)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size//(2 if bid else 1), num_layers, batch_first=True, bidirectional=bid)

        self.init_weights()

    def init_weights(self):
        if self.word_dim > 300:
            self.embed.weight.data.normal_(mean=0, std=0.3785)

    def forward(self, x, lengths, return_x=False):
        """Handles variable size captions"""
        # Embed word ids to vectors
        x_glove = self.glove.index_select(0, x.view(-1)).view(x.size(0), x.size(1), -1)
        if self.word_dim > 300:
            x_semantic = self.embed(x)
            x = torch.cat((x_semantic, x_glove), dim=2)
        else:
            x = x_glove
        if return_x:
            x.requires_grad = True
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True, padding_value=-1e8)
        out = self.combiner(padded[0], lengths)[0]

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        if return_x:
            return out, x
        return out

    def __call__(self, x, lengths):
        return self.forward(x, lengths)


# Bi-LSTM + max pooling, current not implemented other options
class EncoderTextLSTM(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, pooling='last', use_abs=False, bid=False,
                 glove_path='data/glove.pkl'):
        super(EncoderTextLSTM, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.combiner = Combiner(pooling, embed_size)
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim-300)
        _, embed_weight = pickle.load(open(glove_path, 'rb'))
        self.glove = Variable(torch.cuda.FloatTensor(embed_weight), requires_grad=False)
        # caption embedding
        self.rnn = nn.LSTM(word_dim, embed_size//(2 if bid else 1), num_layers, batch_first=True, bidirectional=bid)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0, std=0.3785)

    def forward(self, x, lengths):
        # Embed word ids to vectors
        x_glove = self.glove.index_select(0, x.view(-1)).view(x.size(0), x.size(1), -1)
        x_semantic = self.embed(x)
        x = torch.cat((x_semantic, x_glove), dim=2)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True, padding_value=-1e8)
        out = self.combiner(padded[0], lengths)[0]

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# CNN Text Encoder
class EncoderTextCNN(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, use_abs=False, glove_path='data/glove.pkl'):
        super(EncoderTextCNN, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim-300, padding_idx=0)  # 0 for <pad>
        _, embed_weight = pickle.load(open(glove_path, 'rb'))
        self.glove = Variable(torch.cuda.FloatTensor(embed_weight), requires_grad=False)

        channel_num = embed_size // 4
        self.conv2 = nn.Conv1d(word_dim, channel_num, 2)
        self.conv3 = nn.Conv1d(word_dim, channel_num, 3)
        self.conv4 = nn.Conv1d(word_dim, channel_num, 4)
        self.conv5 = nn.Conv1d(word_dim, channel_num, 5)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

#        self.mlp = nn.Linear(embed_size, embed_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0, std=0.3785)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)
        self.conv5.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        x_glove = self.glove.index_select(0, x.view(-1)).view(x.size(0), x.size(1), -1)
        x_semantic = self.embed(x)
        x = torch.cat((x_semantic, x_glove), dim=2)
        x = torch.transpose(x, 1, 2).contiguous()
        conv2 = self.conv2(x).max(2)[0]
        conv3 = self.conv3(x).max(2)[0]
        conv4 = self.conv4(x).max(2)[0]
        conv5 = self.conv5(x).max(2)[0]
        rep = torch.cat((conv2, conv3, conv4, conv5), dim=1).view(x.size(0), -1)
        # l2-norm
        rep = l2norm(rep)
        if self.use_abs:
            rep = torch.abs(rep)
        return rep

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Deep CNN Text Encoder
class EncoderTextDeepCNN(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, use_abs=False,
                 glove_path='data/glove.pkl'):
        super(EncoderTextDeepCNN, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim-300, padding_idx=0)
        _, embed_weight = pickle.load(open(glove_path, 'rb'))
        self.glove = Variable(torch.cuda.FloatTensor(embed_weight), requires_grad=False)

        channel_num = embed_size

        self.conv1 = nn.Conv1d(word_dim, embed_size, 2, stride=2)   # [batch_size, dim, 30]
        self.conv2 = nn.Conv1d(embed_size, embed_size, 4, stride=2)  # [batch_size, dim, 14]
        self.conv3 = nn.Conv1d(embed_size, embed_size, 5, stride=2)  # [batch_size, dim, 5]
        self.conv4 = nn.Conv1d(embed_size, channel_num, 5)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

#        self.mlp = nn.Linear(embed_size, embed_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0, std=0.3785)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)
        self.conv5.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        x_glove = self.glove.index_select(0, x.view(-1)).view(x.size(0), x.size(1), -1)
        x_semantic = self.embed(x)
        x = torch.cat((x_semantic, x_glove), dim=2)
        x = torch.transpose(x, 1, 2).contiguous()
        x = torch.cat((x, Variable(torch.zeros(x.size(0), x.size(1), 60 - x.size(2)))), dim=2)
        conv1 = functional.relu(self.conv1(x))
        conv2 = functional.relu(self.conv2(conv1))
        conv3 = functional.relu(self.conv3(conv2))
        rep = self.conv4(conv3).view(x.size(0), -1)
        # l2-norm
        rep = l2norm(rep)
        if self.use_abs:
            rep = torch.abs(rep)
        return rep

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Deep CNN Text Encoder
class EncoderTextBoW(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, pooling='attn', use_abs=False):
        super(EncoderTextBoW, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding, high way network
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)  # 0 for <pad>
        self.combiner = Combiner(pooling, embed_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0, std=0.3785)
        self.transfer_fix.weight.data.uniform_(-0.1, 0.1)
        self.transfer_tanh.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        x = self.embed(x)
        rep = self.combiner(x, lengths)[0]
        # l2-norm
        rep = l2norm(rep)
        if self.use_abs:
            rep = torch.abs(rep)
        return rep

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CapOnlyContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(CapOnlyContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s, ex_s):
        # compute image-sentence score matrix
        scores = self.sim(im, ex_s)
        scores_orig = self.sim(im, s)
        diagonal = scores_orig.diag().contiguous().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        return cost_s.sum()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        try:
            if opt.text_encoder_type == 'gru':
                self.txt_enc = EncoderTextGRU(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers,
                                              use_abs=opt.use_abs, bid=opt.bidirectional, pooling=opt.pooling)
            elif opt.text_encoder_type == 'cnn':
                self.txt_enc = EncoderTextCNN(opt.vocab_size, opt.word_dim, opt.embed_size, use_abs=opt.use_abs)
            elif opt.text_encoder_type == 'lstm':
                self.txt_enc = EncoderTextLSTM(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers,
                                               use_abs=opt.use_abs, bid=opt.bidirectional, pooling=opt.pooling)
            elif opt.text_encoder_type == 'deepcnn':
                self.txt_enc = EncoderTextDeepCNN(opt.vocab_size, opt.word_dim, opt.embed_size, use_abs=opt.use_abs)
            elif opt.text_encoder_type == 'bow':
                self.txt_enc = EncoderTextBoW(opt.vocab_size, opt.word_dim, opt.embed_size, pooling=opt.pooling,
                                              use_abs=opt.use_abs)
        except AttributeError:
            self.txt_enc = EncoderTextGRU(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers,
                                          use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(
            margin=opt.margin, measure=opt.measure, max_violation=opt.max_violation)
        self.external_criterion = CapOnlyContrastiveLoss(
            margin=opt.margin, measure=opt.measure, max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.epoch = 0
        self.vocab = None
        self.opt = opt

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images=None, captions=None, lengths=None, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if images is not None:
            images = Variable(images, volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
            img_emb = self.img_enc(images)
        else:
            img_emb = None
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()
            cap_emb = self.txt_enc(captions, lengths)
        else:
            cap_emb = None
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings"""
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def forward_external_loss(self, img_emb, cap_emb, ex_cap_emb, **kwargs):
        """Compute the loss for extended captions, given pairs of image and caption embeddings"""
        loss = self.external_criterion(img_emb, cap_emb, ex_cap_emb)
        self.logger.update('Lex', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

    def train_emb_with_extended_captions(self, images, captions, lengths, ids=None, *args):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # load external captions
        if self.epoch > 0:
            external_captions = list()
            for idx in ids:
                try:
                    lines = open(self.opt.data_path + '/' + self.opt.data_name + '/train_ex/{:d}.txt'.format(
                        idx)).readlines()
                except:
                    print(idx)
                    continue
                random.shuffle(lines)
                if len(lines) > self.opt.negative_number:
                    lines = lines[:self.opt.negative_number]
                lines = [['<start>'] + line.split() + ['<end>'] for line in lines]
                external_captions.extend(lines)
            external_lengths = [len(line) for line in external_captions]
            cap_emb_ex = None
            caps_ex = None
            lengths_ex = None
            for x in range(0, len(external_captions), 2048):
                batch_caption = external_captions[x: min(len(external_captions), x+2048)]
                batch_lengths = external_lengths[x: min(len(external_captions), x+2048)]
                max_length = max(batch_lengths)
                for j in range(len(batch_caption)):
                    batch_caption[j] = [self.vocab.word2idx.get(word, 3) for word in batch_caption[j]]
                    for _ in range(max_length - len(batch_caption[j])):
                        batch_caption[j].append(self.vocab.word2idx['<pad>'])
                batch_caption = Variable(torch.LongTensor(batch_caption), volatile=True)
                batch_lengths = Variable(torch.LongTensor(batch_lengths), volatile=True)
                batch_caption, batch_lengths, inv = sort_sentences_by_lengths(batch_caption, batch_lengths)
                if torch.cuda.is_available():
                    batch_caption = batch_caption.cuda()
                    batch_lengths = batch_lengths.cuda()
                batch_cap_emb = self.txt_enc(batch_caption, batch_lengths.data.tolist())

                if cap_emb_ex is None:
                    cap_emb_ex = batch_cap_emb
                    caps_ex = [batch_caption.data[i].tolist() for i in range(batch_caption.size(0))]
                    lengths_ex = batch_lengths
                else:
                    cap_emb_ex = torch.cat((cap_emb_ex, batch_cap_emb), dim=0)
                    caps_ex = caps_ex + [batch_caption.data[i].tolist() for i in range(batch_caption.size(0))]
                    lengths_ex = torch.cat((lengths_ex, batch_lengths), dim=0)
                scores = cosine_sim(img_emb, cap_emb_ex)
                idxs = scores.max(1)[1]
                cap_emb_ex = cap_emb_ex.index_select(0, idxs)
                caps_ex = [caps_ex[idx] for idx in idxs.data.tolist()]
                lengths_ex = lengths_ex.index_select(0, idxs)
            # compute not volatile embeddings
            max_length = max([len(line) for line in caps_ex])
            for j in range(len(lengths_ex)):
                for _ in range(max_length - len(caps_ex[j])):
                    caps_ex[j].append(self.vocab.word2idx['<pad>'])
            caps_ex = Variable(torch.LongTensor(caps_ex), volatile=False)
            if torch.cuda.is_available():
                caps_ex = caps_ex.cuda()
            lengths_ex = Variable(lengths_ex.data, volatile=False)
            caps_ex, lengths_ex, inv = sort_sentences_by_lengths(caps_ex, lengths_ex)
            cap_emb_ex = self.txt_enc(caps_ex, lengths_ex.data.tolist())
            loss = self.forward_loss(img_emb, cap_emb) + self.forward_external_loss(img_emb, cap_emb, cap_emb_ex)
        else:
            loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


def sort_sentences_by_lengths(x, lengths):
    lengths, idx = lengths.sort(0, descending=True)
    x = x.index_select(0, idx)
    _, inv = idx.sort(0)
    return x, lengths, inv
