from data_loader import *
import argparse
import random
import time
import torch
import numpy as np
import os
from Model.network import GNet
from Model.trainer import Trainer
from torchsummary import summary


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data', type=str, default='synth', help='data folder name')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=3000, help='epochs')
    parser.add_argument('-batch', type=int, default=1, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-n_gcn', type=float, default=3, help='number of gcns')
    parser.add_argument('-n_train_sampl', type=int, default=3, help='number of training samples')
    # parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    # parser.add_argument('-l_num', type=int, default=3, help='layer num')
    # parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=256, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.0, help='drop net')
    # parser.add_argument('-drop_c', type=float, default=1.2, help='drop output')
    parser.add_argument('-act_i', type=str, default='ELU', help='network act input')
    parser.add_argument('-act_o', type=str, default='ELU', help='network act hidden')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act output')
    # parser.add_argument('-weight_decay', type=float, default=0.0008, help='weight decay')
    parser.add_argument('-weight_decay', type=float, default=0.00001, help='weight decay')
    # parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', default=[0.5,0.25,0.125,0.06,0.03, 0.015, 0.0075])
    # parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    print(args)
    
    # get paths
    method_dir =os.path.realpath(os.path.dirname(__file__)) 
    data_dir = method_dir+"/../Dataset/Data/"+args.data
    tables_dir = data_dir+"/table_results"
    checkpoint_dir = method_dir+"/Model/checkpoints"
    geometry_path = data_dir+"/"+args.data+".unv"
    assert(os.path.isdir(tables_dir))
    assert(os.path.isdir(checkpoint_dir))
    assert(os.path.isfile(geometry_path))
    print("tables in ", tables_dir )
    print("geometry: ", geometry_path )

    file_loader = FileLoader(geometry_path, tables_dir, args) # file loader
    data = file_loader.get_data() # graph list
    # data.graph_list[0].debug()
    for fold_idx in range(args.fold):
        print('start training ------> fold', fold_idx+1)
        data.use_fold_data(fold_idx)
        net = GNet(data.n_feas_x, data.n_feas_y, args)
        trainer = Trainer(args, net, data, checkpoint_dir)
        # summary(net, [ 1, GraphData(data.train_gs).__getitem__(0) ] )
        # summary(net, GraphData(data.train_gs).loader(trainer.args.batch, True)  )
        # print(net)
        trainer.train()
        net.predict_and_visualize(data.train_gs[0])
        break # use just 1 fold

if __name__ == "__main__": main()
