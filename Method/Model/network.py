import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ops import GCN, GraphUnet, Initializer, norm_g


class GNet(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.i_act = getattr(nn, args.act_i)()
        self.o_act = getattr(nn, args.act_o)()
        self.i_gcn = GCN(in_dim, args.l_dim, self.i_act, args.drop_n, 1)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n, args.n_gcn)
        self.o_gcn = GCN(args.l_dim, out_dim, self.o_act, args.drop_n, 1)
        # self.o_gcn = GCN(args.l_dim, out_dim, self.n_act, args.drop_n)
        # self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.h_dim)
        # self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        # self.out_drop = nn.Dropout(p=args.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, ys):
        hs, ys = self.embed(gs, hs, ys)
        return self.metric(hs, ys)

    def predict(self, graph):
        gs, hs, ys = graph.graph_data()
        gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
        hs = self.embed_one(gs, hs)
        return hs, ys

    def predict_and_visualize(self, graph):
        hs, ys = self.predict( graph)
        loss = self.metric(hs, ys)
        print(loss)
        hs = hs.cpu()
        # iterate through tensor rows
        pred_list = []
        for row in hs.detach().numpy():
            pred_list.append( row )
        graph.pred_list = pred_list
        graph.debug(groundtruth=True, show=False)
        graph.debug(groundtruth=False, show=True)


    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        # else:
            # print("Cuda NOT available")
            # exit(1)
        return gs

    def embed(self, gs, hs, ys):
        o_hs , o_ys = [], []

        # hs = self.embed_one(gs, hs)
        # o_hs.append(hs)
        # o_ys.append(ys)

        for g, h, y in zip(gs, hs, ys):
            h = self.embed_one(g, h)
            o_hs.append(h)
            o_ys.append(y)

        hs = torch.stack(o_hs, 0)
        ys = torch.stack(o_ys, 0)
        return hs, ys

    def embed_one(self, g, h):
        # print(type(g))s
        # print(g.size())
        g = norm_g(g)
        h = self.i_gcn(g, h)
        h, hs = self.g_unet(g, h)
        h = self.o_gcn(g, h)
        return h

    # def readout(self, hs):
    #     h_max = [torch.max(h, 0)[0] for h in hs]
    #     h_sum = [torch.sum(h, 0) for h in hs]
    #     h_mean = [torch.mean(h, 0) for h in hs]
    #     h = torch.cat(h_max + h_sum + h_mean)
    #     return h



    def metric(self, ys_pred, ys_gt):
        loss = F.mse_loss(ys_pred, ys_gt)
        return loss


    # def metric(self, logits, labels):
    #     loss = F.nll_loss(logits, labels)
    #     _, preds = torch.max(logits, 1)
    #     acc = torch.mean((preds == labels).float())
    #     return loss, acc
