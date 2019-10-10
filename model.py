import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        # reset parameters
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, support, adj):
        support = torch.mm(support, self.weight)
        x = torch.sparse.mm(adj, support)
        if self.bias is None:
            return x
        else:
            return x + self.bias

    def l2_loss(self):
        return self.weight.pow(2).sum()


class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_units, dropout):
        super(Model, self).__init__()
        self.dropout = dropout
        self.gcn_1 = GraphConvLayer(in_features, hidden_units)
        self.gcn_2 = GraphConvLayer(hidden_units, out_features)

    def forward(self, support, adj):
        out = F.relu(self.gcn_1(support, adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gcn_2(out, adj)
        return F.log_softmax(out, dim=1)

    def l2_loss(self):
        return self.gcn_1.l2_loss() + self.gcn_2.l2_loss()
