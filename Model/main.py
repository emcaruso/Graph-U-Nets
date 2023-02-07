from data_loader import *
import argparse
import random
import time
import torch
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='DD', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    # parser.add_argument('-ks', nargs='+', type=float, default='0.9 0.8 0.7')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = get_args()
    print(args)
    data_dir = os.path.realpath(os.path.dirname(__file__)+"/../Dataset/Data")
    tables_dir = data_dir+"/3d_models/table_results/"
    
    file_loader = FileLoader(args)
    # file_loader.load_table(tables_dir+"table_meca.txt")
    # file_loader.load_unv(tables_dir+"Mesh_1.unv")
    file_loader.get_graph(tables_dir+"table_meca.txt",tables_dir+"Mesh_1.unv" )
    print("done")

if __name__ == "__main__":
    main()
