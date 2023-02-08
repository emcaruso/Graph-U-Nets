import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import re



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
                nodes.append({ "id" : node_id, "coords": np.array([node_x, node_y, node_z])})
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
                coords_v1 = next(item for item in nodes if item["id"] == edge_v1)["coords"]
                coords_v2 = next(item for item in nodes if item["id"] == edge_v2)["coords"]
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
        dx_idx = feat_list.index('DX')
        dy_idx = feat_list.index('DY')
        dz_idx = feat_list.index('DZ')
        id_idx = feat_list.index('NOEUD')
        flux_idx = feat_list.index('FLUX')
        fluy_idx = feat_list.index('FLUY')
        fluz_idx = feat_list.index('FLUZ')
        coorx_idx = feat_list.index('COOR_X')
        coory_idx = feat_list.index('COOR_Y')
        coorz_idx = feat_list.index('COOR_Z')
        node_features = []

        # ==== LOAD DICTS

        ordered = False
        for i,line in enumerate(lines[1:]):
            node_list_str = re.sub(' +', ' ', line.strip()).split(" ")
            if node_list_str[flux_idx] == "-":
                node_features.append( 
                        ( int(node_list_str[id_idx][1:]) ,
                            {
                                "id" : int(node_list_str[id_idx][1:]),
                                "dx" : float(node_list_str[dx_idx]),
                                "dy" : float(node_list_str[dy_idx]),
                                "dz" : float(node_list_str[dz_idx]),
                                "coorx" : float(node_list_str[coorx_idx]),
                                "coory" : float(node_list_str[coory_idx]),
                                "coorz" : float(node_list_str[coorz_idx])
                            }
                        ) )
            else:
                if not ordered:
                    node_features = sorted(node_features, key=lambda i: i[0])
                    ordered = True
                    
                id = int(node_list_str[id_idx][1:])
                id_list = id-1
                assert(id == node_features[id_list][1]["id"])
                assert(id == node_features[id_list][0])
                feature = node_features[id_list][1]
                feature["flux"] = float(node_list_str[flux_idx])
                feature["fluy"] = float(node_list_str[fluy_idx])
                feature["fluz"] = float(node_list_str[fluz_idx])


        return node_features

        
    def create_graph(self, node_features, edges ):
        
        graph = nx.Graph()
        graph.add_nodes_from(node_features)
        print(edges)
        graph.add_weighted_edges_from(edges)

        return graph

        # print(list(graph.nodes.data()))
        print(list(graph.edges.data()))
        
    def get_graph(self, table_path, unv_path):

        node_features = self.load_table(table_path)
        nodes, edges = self.load_unv(unv_path)

        graph = self.create_graph( node_features, edges )
        return graph


