from data_loader import *
import argparse
import random
import time
import torch
import numpy as np
import os
from Model.network import GNet
from Model.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    # parser.add_argument('-data', default='DD', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=200, help='epochs')
    # parser.add_argument('-num_epochs_save', type=int, default=20, help='epoch save for history')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    # parser.add_argument('-l_num', type=int, default=3, help='layer num')
    # parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    # parser.add_argument('-drop_c', type=float, default=1.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    # parser.add_argument('weight_decay', type=float, default=0.0008, help='weight decay')
    # parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', default=[0.9, 0.8, 0.7])
    # parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args

def get_filelist():
    data_dir = os.path.realpath(os.path.dirname(__file__)+"/../Dataset/Data")
    tables_dir = data_dir+"/table_results"
    table_names = os.listdir(tables_dir)

def main():
    args = get_args()
    # dictn = {k:vars(args)[k] for k in ('batch','lr','l_dim','drop_n','act_n') if k in vars(args)}
    # args_str = str(dictn)
    print(args)
    get_filelist()
    
    # get paths
    method_dir =os.path.realpath(os.path.dirname(__file__)) 
    data_dir = method_dir+"/../Dataset/Data"
    tables_dir = data_dir+"/table_results"
    table_paths = os.listdir(tables_dir)
    checkpoint_dir = method_dir+"/Model/checkpoints"
    geometry_path = data_dir+"/geometry_cj.unv"
    print("tables in ", tables_dir )
    print("geometry: ", geometry_path )

    file_loader = FileLoader(args) # file loader
    data = file_loader.get_data(tables_dir, geometry_path) # graph list
    for fold_idx in range(args.fold):
        print('start training ------> fold', fold_idx+1)
        data.use_fold_data(fold_idx)
        net = GNet(data.n_feas_x, data.n_feas_y, args)
        trainer = Trainer(args, net, data, checkpoint_dir)
        trainer.train()
        break # use just 1 fold

if __name__ == "__main__":
    main()
