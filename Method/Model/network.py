import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ops import GCN, GraphUnet, Initializer, norm_g


class GNet(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.i_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n)
        self.o_gcn = GCN(args.l_dim, out_dim, torch.nn.modules.activation.ReLU(), args.drop_n)
        # self.o_gcn = GCN(args.l_dim, out_dim, self.n_act, args.drop_n)
        # self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.h_dim)
        # self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        # self.out_drop = nn.Dropout(p=args.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, ys):
        hs, ys = self.embed(gs, hs, ys)
        return self.metric(hs, ys)

    def embed(self, gs, hs, ys):
        o_hs , o_ys = [], []
        for g, h, y in zip(gs, hs, ys):
            h = self.embed_one(g, h)
            o_hs.append(h)
            o_ys.append(y)
        hs = torch.stack(o_hs, 0)
        ys = torch.stack(o_ys, 0)
        return hs, ys

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.i_gcn(g, h)
        hs = self.g_unet(g, h)
        h = self.o_gcn(g, h)
        return h

    # def readout(self, hs):
    #     h_max = [torch.max(h, 0)[0] for h in hs]
    #     h_sum = [torch.sum(h, 0) for h in hs]
    #     h_mean = [torch.mean(h, 0) for h in hs]
    #     h = torch.cat(h_max + h_sum + h_mean)
    #     return h

    # def classify(self, h):
    #     h = self.out_drop(h)
    #     h = self.out_l_1(h)
    #     h = self.c_act(h)
    #     h = self.out_drop(h)
    #     h = self.out_l_2(h)
    #     return F.log_softmax(h, dim=1)


    def metric(self, ys_pred, ys_gt):
        loss = F.mse_loss(ys_pred, ys_gt)
        return loss


    # def metric(self, logits, labels):
    #     loss = F.nll_loss(logits, labels)
    #     _, preds = torch.max(logits, 1)
    #     acc = torch.mean((preds == labels).float())
    #     return loss, acc
