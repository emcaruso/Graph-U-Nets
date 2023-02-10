import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import re

import os


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
                node_x = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[0])
                node_y = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[1])
                node_z = float(re.sub(' +', ' ', lines[line_idx+1].strip()).split(" ")[2])
                if node_id == -1:
                    flag = False
                    line_idx += 3
                    continue
                nodes.append( ( node_id, { "id" : node_id, "coords": np.array([node_x, node_y, node_z])} ) )
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

        print("loading unv file")

        with open(unv_path, 'r') as f:
            lines = f.readlines()

        nodes, edges = self.get_nodes_edges(lines)

        return nodes, edges


    def load_table(self, table_path):

        # ==== READ FILE

        print("loading files")

        with open(table_path, 'r') as f:
            lines = f.readlines()[4:-1]
        
        # ====  LOAD INDICES

        feat_list = re.sub(' +', ' ', lines[0].strip()).split(" ")
        id_idx = feat_list.index('NOEUD')
        node_features = []

        # ==== LOAD DICTS

        ordered = False
        for i,line in enumerate(lines[1:]):
            node_list_str = re.sub(' +', ' ', line.strip()).split(" ")
            if node_list_str[0] == "Displacements":
                node_features.append( 
                            (
                                int(node_list_str[id_idx][1:]),
                                {
                                        "id" : int(node_list_str[id_idx][1:]),
                                        "displacements" : np.array([float(node_list_str[feat_list.index('DX')]),
                                                                    float(node_list_str[feat_list.index('DY')]),
                                                                    float(node_list_str[feat_list.index('DZ')])]),
                                        "coordinates" : np.array([float(node_list_str[feat_list.index('COOR_X')]),
                                                                  float(node_list_str[feat_list.index('COOR_Y')]),
                                                                  float(node_list_str[feat_list.index('COOR_Z')])]),
                                }
                            )
                        )
            else:
                if not ordered:
                    node_features = sorted(node_features, key=lambda i: i[0])
                    ordered = True
                    
                id = int(node_list_str[id_idx][1:])
                id_list = id-1
                feature = node_features[id_list][1]
                feature["flux"] = np.array([float(node_list_str[feat_list.index('FLUX')]),
                                            float(node_list_str[feat_list.index('FLUY')]),
                                            float(node_list_str[feat_list.index('FLUZ')])]),
                


        return node_features

        
    def create_graph(self, node_features, edges):
        
        graph = nx.Graph()
        graph.add_nodes_from(node_features)
        graph.add_weighted_edges_from(edges)

        print(list(graph.nodes.data()))
        # print(list(graph.edges.data()))
        return graph

        
    def get_graph(self, table_path, unv_path):

        node_features = self.load_table(table_path)
        nodes, edges = self.load_unv(unv_path)
        assert( len(node_features)==len(nodes) )

        graph = self.create_graph( node_features, edges)


        return graph


    def get_graphs(self, tables_dir, unv_path):
        graphs = []
        for table_path in os.listdir(tables_dir):
            graph = self.get_graph(tables_dir+"/"+table_path, unv_path)
            graphs.append(graph)
        return graphs

    

