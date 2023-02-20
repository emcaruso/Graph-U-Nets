import torch
import torch.nn as nn
import numpy as np


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p, n_gcn):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p, n_gcn)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p, n_gcn))
            self.up_gcns.append(GCN(dim, dim, act, drop_p, n_gcn))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            # h = h.add(down_outs[up_idx])
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        # return hs
        return h, hs


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p, n_gcn):
        super(GCN, self).__init__()
        self.projs = nn.ModuleList()
        self.act = act
        self.drops = nn.ModuleList()
        self.n_gcn = n_gcn
        for i in range(int(n_gcn)):
            self.projs.append(nn.Linear(in_dim, out_dim))
            self.drops.append( nn.Dropout(p=p) if p > 0.0 else nn.Identity())

    def forward(self, g, h):
        for i in range(int(self.n_gcn)):
            h = self.drops[i](h)
            h = torch.matmul(g, h)
            h = self.projs[i](h)
            h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        # exit()
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.sparse.mm(un_g, un_g).bool().float()
    # un_g = torch.matmul(un_g, un_g).bool().float()
    # un_g = un_g[idx, :]
    un_g = un_g.index_select(0,idx)
    # un_g = un_g[:, idx]
    un_g = un_g.index_select(1,idx)
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sparse.sum(g, 1)
    # degrees = torch.sum(g, 1)
    # print(degrees)
    # g = torch.divide( g, degrees )
    
    inv_degrees = (1.0 / degrees.to_dense())
    if torch.cuda.is_available():
        diag_inv_scalars = torch.sparse_coo_tensor(indices=torch.stack([torch.arange(g.shape[0]), torch.arange(g.shape[0])]).cuda(), values=inv_degrees, size=g.shape).cuda()
    else:
        diag_inv_scalars = torch.sparse_coo_tensor(indices=torch.stack([torch.arange(g.shape[0]), torch.arange(g.shape[0])]), values=inv_degrees, size=g.shape)
    g = torch.sparse.mm(g, diag_inv_scalars)

    # print(g)
    # g = g / degrees
    # print(g.to_dense())
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
