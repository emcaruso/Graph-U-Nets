import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import re
import os
import random


class FileLoader(object):
    def __init__(self, args):
        self.args = args

    def get_nodes_edges(self, lines):
        nodes = []
        edges = []
        flag = True
        line_idx = 19
        max_distance = 0
        
        while True:
            if flag:
                line = lines[line_idx]

                node_id = int(re.sub(' +', ' ', line.strip()).split(" ")[0])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes_x = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[0])
                nodes_y = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[1])
                node_z = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[2])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes.append( ( node_id, { "id" : node_id, "coords": np.array([nodes_x, nodes_y, node_z])} ) )
                line_idx += 2
            else:
                line = lines[line_idx]
                edge_id = int(re.sub(' +', ' ', line.strip()).split(" ")[0])
                edge_type = int(re.sub(' +', ' ', line.strip()).split(" ")[1])
                if edge_type != 11:
                    break

                line_idx += 2

                line = lines[line_idx]
                edge_v1 = int(re.sub(' +', ' ', line.strip()).split(" ")[0])
                edge_v2 = int(re.sub(' +', ' ', line.strip()).split(" ")[1])
                coords_v1 = next(item for item in nodes if item[0] == edge_v1)[1]["coords"]
                coords_v2 = next(item for item in nodes if item[0] == edge_v2)[1]["coords"]
                distance = np.sqrt(np.sum(np.power((coords_v1-coords_v2),2)))
                max_distance = max(distance, max_distance)
                edges.append( (edge_v1, edge_v2, distance) )
                edges.append( (edge_v2, edge_v1, distance) ) # undirected?

                line_idx += 1

        edges = list(map(lambda t : (t[0],t[1],t[2]/max_distance), edges))
        return nodes, edges

        
    def load_unv(self, unv_path):
        
        # ==== READ FILE

        print("Loading unv file ...")

        with open(unv_path, 'r') as f:
            lines = f.readlines()

        nodes, edges = self.get_nodes_edges(lines)

        return nodes, edges


    def load_table(self, table_path):

        # ==== READ FILE

        # print("loading unv: ", table_path)

        with open(table_path, 'r') as f:
            lines = f.readlines()[4:-1]
        
        # ====  LOAD INDICES

        feat_list = re.sub(' +', ' ', lines[0].strip()).split(" ")
        id_idx = feat_list.index('NOEUD')
        nodes_x = []
        nodes_y = []

        # ==== LOAD DICTS

        ordered = False
        for i,line in enumerate(lines[1:]):
            node_list_str = re.sub(' +', ' ', line.strip()).split(" ")
            if node_list_str[0] == "Displacements":
                
                nodes_x.append( 
                        (
                                int(node_list_str[id_idx][1:]),
                                {
                                        # "id" : int(node_list_str[id_idx][1:]),
                                        # "cx" : float(node_list_str[feat_list.index("COOR_X")]),
                                        # "cy" : float(node_list_str[feat_list.index("COOR_Y")]),
                                        # "cz" : float(node_list_str[feat_list.index("COOR_Z")]),
                                        "dx" : float(node_list_str[feat_list.index("DX")]),
                                        "dy" : float(node_list_str[feat_list.index("DY")]),
                                        "dz" : float(node_list_str[feat_list.index("DZ")])
                                        # "displacements" : np.array([float(node_list_str[feat_list.index('DX')]),
                                        #                             float(node_list_str[feat_list.index('DY')]),
                                        #                             float(node_list_str[feat_list.index('DZ')])]),
                                        # "coordinates" : np.array([float(node_list_str[feat_list.index('COOR_X')]),
                                        #                           float(node_list_str[feat_list.index('COOR_Y')]),
                                        #                           float(node_list_str[feat_list.index('COOR_Z')])]),
                                }
                        ))
                nodes_y.append(
                        (
                                int(node_list_str[id_idx][1:]),
                                {
                                        # "id" : int(node_list_str[id_idx][1:]),
                                }
                        ))

            else:
                if not ordered:
                    nodes_x = sorted(nodes_x, key=lambda i: i[0])
                    nodes_y = sorted(nodes_y, key=lambda i: i[0])
                    ordered = True
                    
                id = int(node_list_str[id_idx][1:])
                id_list = id-1
                feature = nodes_y[id_list][1]
                feature["flux"] = float(node_list_str[feat_list.index('FLUX')])
                feature["fluy"] = float(node_list_str[feat_list.index('FLUY')])
                feature["fluz"] = float(node_list_str[feat_list.index('FLUZ')])
                # feature["flux"] = np.array([float(node_list_str[feat_list.index('FLUX')]),
                #                             float(node_list_str[feat_list.index('FLUY')]),
                #                             float(node_list_str[feat_list.index('FLUZ')])]),

        return nodes_x, nodes_y

        
    def create_graph(self, nodes_x, edges):
        graph = nx.Graph()
        graph.add_nodes_from(nodes_x)
        graph.add_weighted_edges_from(edges)
        return graph
        

    def nodes_2_feas(self, nodes):
        feas = [ list(item[1].values()) for item in nodes]
        return torch.tensor(feas)


    def get_graph(self, table_path, nodes, edges):
        nodes_x, nodes_y = self.load_table(table_path)
        assert( len(nodes_x)==len(nodes) )
        assert( len(nodes_y)==len(nodes) )

        feas_x = self.nodes_2_feas(nodes_x)
        feas_y = self.nodes_2_feas(nodes_y)

        graph =self.create_graph( nodes_x, edges)
        graph.feas_x = feas_x
        graph.feas_y = feas_y

        A = torch.FloatTensor(nx.to_numpy_matrix(graph))
        graph.A = A + torch.eye(graph.number_of_nodes())
        return graph

    def get_graphs(self, tables_dir, nodes, edges):
        graphs = []
        for table_path in tqdm(os.listdir(tables_dir), desc="Loading tables", unit='graphs'):
            graph = self.get_graph(tables_dir+"/"+table_path, nodes, edges)
            graphs.append(graph)
        return graphs

    def get_data(self, tables_dir, unv_path):
        nodes, edges = self.load_unv(unv_path)
        graphs = self.get_graphs(tables_dir, nodes, edges)
        return G_data( graphs )
    
class G_data(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.sep_data()
        self.n_nodes = self.get_n_nodes()
        self.n_feas_x = self.get_n_feas_x()
        self.n_feas_y = self.get_n_feas_y()

    def get_n_nodes(self):
        n_nodes = len(self.graphs[0].nodes)
        for g in self.graphs:
            assert(len(g.nodes)==n_nodes)
        return n_nodes

    def get_n_feas_x(self):
        n_features = self.graphs[0].feas_x.size(dim=1)
        for g in self.graphs:
            assert(g.feas_x.size(dim=1)==n_features)
        return n_features

    def get_n_feas_y(self):
        n_features = self.graphs[0].feas_y.size(dim=1)
        for g in self.graphs:
            assert(g.feas_y.size(dim=1)==n_features)
        return n_features

    def sep_data(self, seed=0):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        labels = [0] * len(self.graphs)
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx+1
        train_idx, test_idx = self.idx_list[fold_idx]
        self.train_gs = [ self.graphs[i] for i in train_idx]
        self.test_gs =  [ self.graphs[i] for i in test_idx]

class GraphData(object):

    def __init__(self, data):
        super(GraphData, self).__init__()
        self.data = data
        self.idx = list(range(len(data)))
        self.pos = 0

    def __reset__(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        return len(self.data) // self.batch + 1

    def __getitem__(self, idx):
        g = self.data[idx]
        return g.A, g.feas_x.float(), g.feas_y.float()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.data):
            self.__reset__()
            raise StopIteration

        cur_idx = self.idx[self.pos: self.pos+self.batch]
        data = [self.__getitem__(idx) for idx in cur_idx]
        self.pos += len(cur_idx)
        gs, xs, ys = map(list, zip(*data))
        return len(gs), gs, xs, ys

    def loader(self, batch, shuffle, *args):
        self.batch = batch
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.idx)
        return self
