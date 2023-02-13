import torch
from tqdm import tqdm
import torch.optim as optim
from data_loader import GraphData


class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.test_gs)
        if torch.cuda.is_available():
            self.net.cuda()

    def init(self, args, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        train_data = GraphData(train_gs)
        test_data = GraphData(test_gs)
        self.train_d = train_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer):
        losses, n_samples = [], 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, xs, ys = batch
            gs, xs, ys = map(self.to_cuda, [gs, xs, ys])
            loss = model(gs, xs, ys)
            losses.append(loss*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = sum(losses) / n_samples
        return avg_loss.item()

    def train(self):
        train_str = 'Train epoch %d: loss %.5f'
        test_str = 'Test epoch %d: loss %.5f'
        line_str = '%d:\t%.5f\n'
        for e_id in range(self.args.num_epochs):
            self.net.train()
            loss = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss))

            with torch.no_grad():
                self.net.eval()
                loss= self.run_epoch(e_id, self.test_d, self.net, None)
            print(test_str % (e_id, loss))

        # with open(self.args.acc_file, 'a+') as f:
        #     f.write(line_str % (self.fold_idx, max_acc))
