import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import re

class DDDataset_txt(object):
    
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.file_exists()
        sparse_adjacency, node_features, node_labels, graph_num, graph_indicator = self.read_data()
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_features = node_features
        self.node_labels = node_labels
        self.graph_num = graph_num
        self.graph_indicator = graph_indicator
        
 
    def split_data(self, train_size=0.9, random_num=2024):
        unique_indicator = np.arange(len(self.graph_indicator),dtype=np.int64)
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=random_num)
        return train_index, test_index
    
    def __getitem__(self, index):
        mask = np.isin(self.graph_indicator,index)
        graph_indicator = self.graph_indicator[mask]
        map_item = np.array([np.unique(graph_indicator),np.arange(len(set(graph_indicator)))]).T
        map_dict = dict(map_item)
        graph_indicator = np.vectorize(map_dict.get)(graph_indicator)
        node_features = self.node_features[mask]
        node_labels = self.node_labels[mask]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_features, graph_indicator, node_labels
    
    def __len__(self):
        return len(self.node_features)
    
    def read_data(self):
        data_dir = self.data_root
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "node_connect.txt"),
                                       dtype=np.int64, delimiter=',')
        node_features = np.genfromtxt(os.path.join(data_dir, "node_features.txt"), 
                                    dtype=np.int64)
        node_labels = np.genfromtxt(os.path.join(data_dir, "node_labels.txt"), 
                                     dtype=np.int64)
        graph_num = np.genfromtxt(os.path.join(data_dir, "graph_num.txt")
                                  ,dtype=str,delimiter=' : ',encoding='gbk')
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "graph_indicator.txt")
                                        ,dtype=np.int64)
        num_nodes = len(node_features)
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)), 
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        return sparse_adjacency, node_features, node_labels, graph_num, graph_indicator
    
    def file_exists(self):
        if not os.path.exists(self.data_root):
            return 0
        else:
            for dirpath,dirnames,filenames in os.walk(self.data_root):
                if 'node_indicator.txt' in filenames and 'node_features.txt' in filenames and \
                    'node_connect.txt' in filenames and 'node_labels.txt' in filenames:
                    return 1
                else:
                    return 0
                
    def leaveone_split(self,name):
        graph_num = self.graph_num
        if graph_num.ndim == 1:
            graph_num = graph_num[np.newaxis, :]
        train_list = []
        test_list = []
        for i in range(graph_num.shape[0]):
            if name in graph_num[i][1]:
                test_list.append(i)
            else:
                train_list.append(i)
        test_graph_indicator_mask = np.isin(self.graph_indicator,test_list)
        train_graph_indicator_mask = ~ test_graph_indicator_mask
        return train_list, test_list, train_graph_indicator_mask, test_graph_indicator_mask
    

