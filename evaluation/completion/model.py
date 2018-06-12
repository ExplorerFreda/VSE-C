import itertools

import torch
import torch.nn as nn
import torch.nn.init as init

from jacinle.utils.container import GView
from jactorch.functional.indexing import index_one_hot_ellipsis
from jactorch.functional.kernel import cosine_distance
from jactorch.functional.linalg import normalize
from jactorch.nn.rnn_utils import rnn_with_length
from jactorch.graph.variable import var_with
from jactorch.quickstart.models import MLPModel


def cosine(input1, input2):
    input1 = normalize(input1, eps=1e-6)
    input2 = normalize(input2, eps=1e-6)
    return (input1 * input2).sum(dim=-1)


def cosine_loss(input1, input2):
    return 1 - cosine(input1, input2)


class CompletionModel(nn.Module):
    def __init__(self, embedding):
        super(CompletionModel, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.hidden_dim = 512
        self.image_dim = 2048
        self.gru_f = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=False)
        self.gru_b = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=False)
        self.predict = MLPModel(self.hidden_dim * 2 + self.image_dim, self.embedding_dim, [], activation='relu')
        self.init_weights()

    def init_weights(self):
        for name, parameter in itertools.chain(self.gru_f.named_parameters(), self.gru_b.named_parameters()):
            if name.startswith('weight'):
                init.orthogonal(parameter.data)
            elif name.startswith('bias'):
                parameter.data.zero_()
            else:
                raise ValueError('Unknown parameter type: {}'.format(name))

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        feature_f = self._extract_sent_feature(feed_dict.sent_f, feed_dict.sent_f_length, self.gru_f)
        feature_b = self._extract_sent_feature(feed_dict.sent_b, feed_dict.sent_b_length, self.gru_b)
        feature_img = feed_dict.image
        
        feature = torch.cat([feature_f, feature_b, feature_img], dim=1)
        predict = self.predict(feature)

        if self.training:
            label = self.embedding(feed_dict.label)
            loss = cosine_loss(predict, label).mean()
            return loss, {}, {}
        else:
            output_dict = dict(pred=predict)
            if 'label' in feed_dict:
                dis = cosine_distance(predict, self.embedding.weight)
                _, topk = dis.topk(1000, dim=1, sorted=True)
                for k in [1, 10, 100, 1000]:
                    output_dict['top{}'.format(k)] = torch.eq(topk, feed_dict.label.unsqueeze(-1))[:, :k].float().sum(dim=1).mean()
            return output_dict

    def _extract_sent_feature(self, sent, length, gru):
        sent = self.embedding(sent)
        batch_size = sent.size(0)

        state_shape = (1, batch_size, self.hidden_dim)
        initial_state = var_with(torch.zeros(state_shape), sent)
        rnn_output, _ = rnn_with_length(gru, sent, length, initial_state)
        rnn_result = index_one_hot_ellipsis(rnn_output, 1, length - 1)

        return rnn_result

