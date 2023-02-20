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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Graphs(object):
    def __init__(self, graph_list, n_train_sampl):
        self.n_train_sampl = n_train_sampl
        self.graph_list = graph_list
        self.max_x, self.max_y = self.get_maxs()
        self.n_feas_x = self.graph_list[0].feas_x_torch.size(dim=1)
        self.n_feas_y = self.graph_list[0].feas_y_torch.size(dim=1)
        self.n_nodes = len(self.graph_list[0].graph.nodes)
        self.idx_list = self.sep_data()

    def get_maxs(self):
        maxs_x = [graph.max_x for graph in self.graph_list ]
        maxs_y = [graph.max_y for graph in self.graph_list ]
        return max(maxs_x), max(maxs_y)

    def normalize(self):
        self.get_maxs()
        print(f"{ self.max_x = }")
        print(f"{ self.max_y = }")
        for graph in self.graph_list:
            graph.feas_x_torch = torch.mul(graph.feas_x_torch, 1.0/self.max_x)
            graph.feas_y_torch = torch.mul(graph.feas_y_torch, 1.0/self.max_y)
       
    def sep_data(self, seed=0):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = list(skf.split(np.zeros(len(self.graph_list)), [0] * len(self.graph_list )))
        return idx_list

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx+1
        train_idx, test_idx = self.idx_list[fold_idx]
        self.test_gs =  [ self.graph_list[i] for i in test_idx]
        self.train_gs = [ self.graph_list[i] for i in train_idx]
        if self.n_train_sampl > 0:
            self.train_gs = self.train_gs[:self.n_train_sampl]


class Graph(object):
    def __init__(self, node_coords, edges, feas_x, feas_y, name):
        self.name = name
        self.node_coords = node_coords
        self.edges = edges
        self.feas_x = feas_x
        self.feas_y = feas_y
        self.feas_x_torch, self.max_x = self.feas_2_feastorch(feas_x)
        self.feas_y_torch, self.max_y = self.feas_2_feastorch(feas_y)
        
        self.graph = self.create_graph() 
        self.A = self.get_A()
        self.pred_list = None

    
    def graph_data(self):
        return self.A, self.feas_x_torch.float(), self.feas_y_torch.float()


    def get_A(self):
        Acoo = nx.to_scipy_sparse_array(self.graph).tocoo() 
        A = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.FloatTensor(Acoo.data.astype(np.float32))) 
        # A =torch.FloatTensor(nx.to_numpy_array(graph))
        return A

    def feas_2_feastorch(self, feas):
        feas_torch = [ list(item[1].values()) for item in feas]
        feas_torch = torch.tensor(feas_torch)
        max_val = feas_torch.max()
        return feas_torch, max_val

    def create_graph(self):
        # assert( len(feas_x)==len(nodes) )
        graph = nx.Graph()
        graph.add_nodes_from(self.feas_x)
        graph.add_weighted_edges_from(self.edges)
        return graph

    def debug(self, groundtruth=True, show_edges=False, show=True):
        pos = {}

        # # show displacements
        # for v in graph.nodes.data():
        #     pos[v[0]] = np.array([v[1]["dx"],v[1]["dy"],v[1]["dz"]])

        # coordinates
        pos = [ coord[1]["coords"] for coord in self.node_coords if coord[0] in list(map(lambda x : x[0], self.feas_x)) ]

        color_map = []
        if groundtruth:
            color_map = [ y[1]["fluxn"] for y in self.feas_y ]
        else:
            assert self.pred_list is not None
            color_map = [ pred for pred in self.pred_list ]

        node_xyz = np.array([ v for v in pos])
        if show_edges:
            edge_xyz = np.array( [ np.array([node_xyz[e[0]-1],node_xyz[e[1]-1]])  for e in self.graph.edges()] )
        else:
            edge_xyz = np.array( [])
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w", c=color_map)
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")
        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        _format_axes(ax)
        fig.tight_layout()
        plt.title(self.name)

        if(show):
            plt.show()

class FileLoader(object):
    def __init__(self, unv_path, tables_dir, args):
        self.args = args
        self.unv_path = unv_path    
        self.tables_dir = tables_dir
        self.max_disp = 0
        self.max_flux = 0

    def get_nodes_edges(self, lines, volume_flag=False):
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
                feas_x = float(lines[line_idx+1][0])
                feas_y = float(lines[line_idx+1][1])
                node_z = float(lines[line_idx+1][2])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes.append( ( node_id, { "id" : node_id, "coords": np.array([feas_x, feas_y, node_z])} ) )
                # edges.append( (node_id, node_id, 1) )
                line_idx += 2
            else:
                # break

                line = lines[line_idx]
                if line[0]=='-1':
                    break
                el_type = int(line[1])
                if el_type == 11:
                    if not volume_flag:
                        line_curr = lines[line_idx+2]
                        vi = int(line_curr[0])
                        vj = int(line_curr[1])
                        coords_vi = [item for item in nodes if item[0] == vi]
                        coords_vi = next(item for item in nodes if item[0] == vi)[1]["coords"]
                        coords_vj = next(item for item in nodes if item[0] == vj)[1]["coords"]
                        distance = np.sqrt(np.sum(np.power((coords_vi-coords_vj),2)))
                        max_distance = max(distance, max_distance)
                        # edges.append( (vi, vj, distance) )
                        # edges.append( (vj, vi, distance) ) # undirected?
                        edges.append( (vi, vj, distance) )
                        edges.append( (vj, vi, distance) ) # undirected?
                        edges.append( (vi, vi, 0) )
                        edges.append( (vj, vj, 0) )
                    line_idx += 3

                elif el_type == 41:
                    if not volume_flag:
                        break
                    line_idx += 2
                elif el_type == 111 and volume_flag:
                    line = lines[line_idx+1]
                    vs = list(map(int, line[0:4]))
                    for vi in vs:
                        for vj in vs:
                            coords_vi = next(item for item in nodes if item[0] == vi)[1]["coords"]
                            coords_vj = next(item for item in nodes if item[0] == vj)[1]["coords"]
                            distance = np.sqrt(np.sum(np.power((coords_vi-coords_vj),2)))
                            max_distance = max(distance, max_distance)
                            edges.append( (vi, vj, distance) )
                            edges.append( (vj, vi, distance) ) # undirected?
                    line_idx += 2
                else:

                    break

        # edges = list(map(lambda t : (t[0],t[1],1-t[2]/max_distance), edges))
        edges = [ (e[0],e[1],1-(e[2]/max_distance)) for e in edges ]


        return nodes, edges        

    def load_unv(self):
        
        # ==== READ FILE

        print("Loading unv file ...")

        with open(self.unv_path, 'r') as f:
            lines = f.readlines()

        volume_name = self.unv_path.split("/")[-2][-4:]
        volume_flag = volume_name == "_vol"
        nodes, edges  = self.get_nodes_edges(lines, volume_flag)

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
        feas_x = []
        feas_y = []

        # ==== Compute features

        ordered = False
        for i,line in enumerate(lines[1:]):
            node_list_str = re.sub(' +', ' ', line.strip()).split(" ")
            if node_list_str[0] == "Displacements":
                
                feas_x.append( 
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
                feas_y.append(
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
                    feas_x = sorted(feas_x, key=lambda i: i[0])
                    feas_y = sorted(feas_y, key=lambda i: i[0])
                    ordered = True
                    
                id = int(node_list_str[id_idx][1:])
                id_list = id-1
                feature = feas_y[id_list][1]
                fx = float(node_list_str[feat_list.index('FLUX')])
                fy = float(node_list_str[feat_list.index('FLUY')])
                fz = float(node_list_str[feat_list.index('FLUZ')])
                feature["fluxn"] = np.sqrt(np.sum(np.power(np.array([fx,fy,fz]),2)))

        return feas_x, feas_y



    def get_data(self):
        nodes, edges  = self.load_unv()
        graph_list = []
        for table_name in tqdm(os.listdir(self.tables_dir), desc="Loading tables", unit='graphs'):
            feas_x, feas_y = self.table2nodefeas(self.tables_dir+"/"+table_name)
            graph = Graph(nodes,edges,feas_x,feas_y, table_name)
            graph_list.append(graph)

        graphs = Graphs(graph_list, self.args.n_train_sampl)
        graphs.normalize()
        return graphs
    
# class G_data(object):
#     def __init__(self, graphs):
#         self.graphs = graphs
#         self.sep_data()
#         self.n_nodes = self.get_n_nodes()
#         self.n_feas_x = self.get_n_feas_x()
#         self.n_feas_y = self.get_n_feas_y()

#     def get_n_nodes(self):
#         n_nodes = len(self.graphs[0].nodes)
#         for g in self.graphs:
#             assert(len(g.nodes)==n_nodes)
#         return n_nodes

#     def get_n_feas_x(self):
#         n_features = self.graphs[0].feas_x.size(dim=1)
#         for g in self.graphs:
#             assert(g.feas_x.size(dim=1)==n_features)
#         return n_features

#     def get_n_feas_y(self):
#         n_features = self.graphs[0].feas_y.size(dim=1)
#         for g in self.graphs:
#             assert(g.feas_y.size(dim=1)==n_features)
#         return n_features

#     def sep_data(self, seed=0):
#         skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#         labels = [0] * len(self.graphs)
#         self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

#     def use_fold_data(self, fold_idx):
#         self.fold_idx = fold_idx+1
#         train_idx, test_idx = self.idx_list[fold_idx]
#         train_gs = [ self.graphs[i] for i in train_idx]
#         test_gs =  [ self.graphs[i] for i in test_idx]
#         return train_gs, test_gs

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
        return g.A, g.feas_x_torch.float(), g.feas_y_torch.float()

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
