"""
模型
"""
import math
import torch
import torch.nn.functional as F

from torch import nn
from config import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        self.gc1 = GraphConvolution(HIDDEN_DIM, HIDDEN_DIM * 2)
        self.gc2 = GraphConvolution(HIDDEN_DIM * 2, OUTPUT_DIM)

    def get_lstm_feature(self, inputs):
        feature_list = []
        for input in inputs:
            input = torch.tensor(input).to(DEVICE)
            out = self.embed(input)
            _, (out, _) = self.lstm(out)
            feature_list.append(out)

        return torch.cat(feature_list, dim=0)

    def forward(self, inputs, adj):
        feature = self.get_lstm_feature(inputs)
        adj = torch.tensor(adj, dtype=torch.float).to(DEVICE)
        out = F.relu(self.gc1(feature, adj))
        out = F.log_softmax(self.gc2(out, adj), dim=1)

        return out


if __name__ == '__main__':
    model = Model()
    inputs = [[0, 1, 2, 3, 4], [1, 2, 3]]
    res = model(inputs, None)
    print(res.shape)
