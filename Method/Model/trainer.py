import torch
from tqdm import tqdm
import torch.optim as optim
from data_loader import GraphData
import os
import pickle
from matplotlib import pyplot as plt

class LossHist:
    def __init__(self, path, name):
        self.losses_train = []
        self.losses_val = []
        self.name = name
        self.path = path

    def save(self):
        with open(self.path+"/"+self.name+'.pkl', 'wb') as f:
            pickle.dump( [self.losses_train,self.losses_val], f)

    def load(self):
        with open(self.path+"/"+self.name+'.pkl', 'rb') as f:
            self.losses = pickle.load(f)

    def plot(self):
        plt.plot(self.losses_train)
        plt.plot(self.losses_val)
        plt.savefig(self.path+"/"+self.name+'.png')

    def append(self,loss_train, loss_val):
        self.losses_train.append(loss_train)
        self.losses_val.append(loss_val)



class Trainer:
    def __init__(self, args, net, G_data, checkpoint_dir):
        self.args = args
        self.net = net
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.test_gs)
        self.epoch = 0
        self.best_loss = 10000000000
        self.args_name = {k:vars(args)[k] for k in ('batch','lr','l_dim','drop_n','act_i','act_n') if k in vars(args)}
        self.args_name = str(self.args_name)[2:-1]
        # self.checkpoint_dir = checkpoint_dir+"/"+str(self.args_name)[2:-1]
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.loss_hist = LossHist(self.checkpoint_dir+"/loss_images", self.args_name)
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
            # weight_decay=self.args.weight_decay)
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        else:
            print("Cuda NOT available")
            exit(1)
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
        losses = []
        count = 0
        for e_id in range(self.epoch,self.args.num_epochs):
            # early stopping
            if count > 10:
                break
            self.epoch = e_id
            self.net.train()
            loss_train = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss_train))

            loss_val = 0
            with torch.no_grad():
                self.net.eval()
                loss_val= self.run_epoch(e_id, self.test_d, self.net, None)
            print(test_str % (e_id, loss_val))
            
            self.loss_hist.append(loss_train,loss_val)
            # if e_id%self.args.num_epochs_save==0 and e_id!=0:

            # save best model
            count += 1
            if self.best_loss>loss_val:
                self.save_last_model(loss_val, self.checkpoint_dir+"/best_models/"+self.args_name)
                count = 0

            # self.loss_hist.save()

        self.loss_hist.plot()

        # with open(self.args.acc_file, 'a+') as f:
        #     f.write(line_str % (self.fold_idx, max_acc))
    def save_last_model(self, loss, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, path )


