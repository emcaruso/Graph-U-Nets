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
from torch_geometric.utils.convert import from_networkx



class FileLoader(object):
    def __init__(self, unv_path, tables_dir, args):
        self.args = args
        self.unv_path = unv_path    
        self.tables_dir = tables_dir
        self.max_disp = 0
        self.max_flux = 0
        self.patches = []

    # def get_patches(self, lines):
        # for i in range(10):
        #     patch_name = "patch_"+str(i)
        #     list = [i for i, line in enumerate(lines) if patch_name in line]
        #     idx = list[0]
        #     j = 1
        #     face_ids = []
        #     node_ids = []
        #     while True:
        #         line = lines[idx+j]
        #         j += 1
        #         if int(line[0]) != 8: break
        #         face_id = [line[1]]
        #         if len(line)>=6: node_ids.append(int(line[5]))
        #         line_nodes = [ for line in lines if line[0] == face_id]
        #         node_ids+= node_ids_curr
        #     self.patches.append(node_ids)

    def get_nodes_edges(self, lines):
        nodes = []
        edges = []
        flag = True
        line_idx = 19
        max_distance = 0

        lines= [ re.sub(' +', ' ', line.strip()).split(" ") for line in lines ]
        
        while True:
            if flag:
                line = lines[line_idx]

                node_id = int(line[0])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes_x = float(lines[line_idx+1][0])
                nodes_y = float(lines[line_idx+1][1])
                node_z = float(lines[line_idx+1][2])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes.append( ( node_id, { "id" : node_id, "coords": np.array([nodes_x, nodes_y, node_z])} ) )
                edges.append( (node_id, node_id, 0) )
                line_idx += 2
            else:
                line = lines[line_idx]
                edge_id = int(line[0])
                edge_type = int(line[1])
                if edge_type != 11:
                    break

                line_idx += 2

                line = lines[line_idx]
                edge_v1 = int(line[0])
                edge_v2 = int(line[1])
                coords_v1 = next(item for item in nodes if item[0] == edge_v1)[1]["coords"]
                coords_v2 = next(item for item in nodes if item[0] == edge_v2)[1]["coords"]
                distance = np.sqrt(np.sum(np.power((coords_v1-coords_v2),2)))
                max_distance = max(distance, max_distance)
                edges.append( (edge_v1, edge_v2, distance) )
                edges.append( (edge_v2, edge_v1, distance) ) # undirected?

                line_idx += 1

        edges = list(map(lambda t : (t[0],t[1],1-t[2]/max_distance), edges))

        # self.get_patches(lines)

        return nodes, edges        

    def load_unv(self):
        
        # ==== READ FILE

        print("Loading unv file ...")

        with open(self.unv_path, 'r') as f:
            lines = f.readlines()

        nodes, edges  = self.get_nodes_edges(lines)

        return nodes, edges

    def table2nodefeas(self, table_path):


        lst = table_path.split("/")[-1].split(",")
        fluxn_val = lst[0]
        patch_idx = int(lst[1][-5])

        with open(table_path, 'r') as f:
            lines = f.readlines()[4:-1]
        
        # ====  LOAD INDICES

        feat_list = re.sub(' +', ' ', lines[0].strip()).split(" ")
        id_idx = feat_list.index('NOEUD')
        nodes_x = []
        nodes_y = []

        # ==== Compute features

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
                                        # "fluxn" : 0.0
                                }
                        ))
            # else:
            #     break

            else:
                if not ordered:
                    nodes_x = sorted(nodes_x, key=lambda i: i[0])
                    nodes_y = sorted(nodes_y, key=lambda i: i[0])
                    ordered = True
                    
                id = int(node_list_str[id_idx][1:])
                id_list = id-1
                feature = nodes_y[id_list][1]
                fx = float(node_list_str[feat_list.index('FLUX')])
                fy = float(node_list_str[feat_list.index('FLUY')])
                fz = float(node_list_str[feat_list.index('FLUZ')])
                feature["fluxn"] = np.sqrt(np.sum(np.power(np.array([fx,fy,fz]),2)))

        # nodes_x = sorted(nodes_x, key=lambda i: i[0])
        # nodes_y = sorted(nodes_y, key=lambda i: i[0])
        # for id in self.patches[patch_idx]:
        #     try:
        #         nodes_y[id][1]["fluxn"]=fluxn_val
        #     except:
        #         print(id, len(nodes_y))
        #         exit(1)


        return nodes_x, nodes_y

    def get_node_boundaries(self, nodes_y):
        print(self.unv_path)
        unv_name = self.unv_path.split("/")[-1][:-4]
        print(unv_name)
        exit(1)

        

        with open(self.unv_path, 'r') as f:
            lines = f.readlines()

        # for line in lines:

        #     if line==


    def create_graph(self, nodes_x, edges):
        graph = nx.Graph()
        graph.add_nodes_from(nodes_x)
        graph.add_weighted_edges_from(edges)
        return graph
        

    def nodes_2_feas(self, nodes):
        feas = [ list(item[1].values()) for item in nodes]
        feas_tens = torch.tensor(feas)
        max_val = feas_tens.max()
        return feas_tens, max_val

    def normalize_graphs(self, graphs):
        print(f"{ self.max_disp = }")
        print(f"{ self.max_flux = }")
        for graph in graphs:
            graph.feas_x = torch.mul(graph.feas_x, 1.0/self.max_disp)
            graph.feas_y = torch.mul(graph.feas_y, 1.0/self.max_flux)
        return graphs


    def get_graph(self, table_path, nodes, edges ):
        nodes_x, nodes_y = self.table2nodefeas(table_path)
        assert( len(nodes_x)==len(nodes) )
        assert( len(nodes_y)==len(nodes) )

        feas_x , norm_x = self.nodes_2_feas(nodes_x)
        feas_y , norm_y = self.nodes_2_feas(nodes_y)

        self.max_disp=max(self.max_disp, norm_x)
        self.max_flux=max(self.max_flux, norm_y)

        graph =self.create_graph( nodes_x, edges)
        graph.feas_x = feas_x
        graph.feas_y = feas_y

        # graph.A = from_networkx(graph)
        # print(graph.A.edge_weight)
        # print(graph.A.edge_weight())

        Acoo = nx.to_scipy_sparse_array(graph).tocoo() 
        graph.A = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.FloatTensor(Acoo.data.astype(np.float32))) 

        # graph.A =torch.FloatTensor(nx.to_numpy_array(graph))

        # graph.A = from_networkx(graph)
        # print(type(graph.A))
        # print(graph.A)
        # exit(1)
        # graph.A = A + torch.eye(graph.number_of_nodes()) # self edges added
        return graph

    def get_graphs(self, nodes, edges ):
        graphs = []
        for table_name in tqdm(os.listdir(self.tables_dir), desc="Loading tables", unit='graphs'):
            graph = self.get_graph(self.tables_dir+"/"+table_name, nodes, edges )
            graphs.append(graph)
        graphs = self.normalize_graphs(graphs)
        return graphs


    def get_data(self):
        nodes, edges  = self.load_unv()
        graphs = self.get_graphs( nodes, edges )
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
