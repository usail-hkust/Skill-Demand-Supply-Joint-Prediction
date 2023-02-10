import torch
import pandas as pd
import os 
import networkx as nx


import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader

from data_loader.utils import _generate_data



class originDataset(Dataset):
     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset: pd.DataFrame
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class SingleSkillDataset(Dataset):

    def __init__(self, data, max_length, graph, *args, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.data = data
        self.graph = graph
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        d_x, s_x = sample['X']
        d_y, s_y = sample['Y_Label']
        t_s = sample['Sta']
        t_e = sample['End']
        l = sample['L']
        s = sample['S']
        # padding
        d_x_padded = F.pad(input=d_x, pad=(t_s, self.max_length - l - t_s))
        s_x_padded = F.pad(input=s_x, pad=(t_s, self.max_length - l - t_s))
        return (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e

class SkillflowDataset(SingleSkillDataset):
#     data dir - ./data/skill_inflow_outflow
#     graph_dir = ./data/myGraph.gpickle
    
    def __init__(self, data_dir, graph_dir, batch_size, max_length, shuffle=True, validation_split=0.0, class_num=10, min_length=6, num_workers=1, training=True):
        self.data_dir = data_dir
        self.datasetinflow =  pd.read_csv(
            os.path.join(data_dir, "skill_inflow_list.csv"),
            encoding='utf-8',
            header=0,
            on_bad_lines='warn',
            index_col=False
        ).T
        self.datasetinflow.columns = self.datasetinflow.loc["time_attr"]
        self.datasetinflow.drop("time_attr", axis=0, inplace=True)
        self.datasetoutflow =  pd.read_csv(
            os.path.join(data_dir, "skill_outflow_list.csv"),
            encoding='utf-8',
            header=0,
            on_bad_lines='warn',
            index_col=False
        ).T
        self.datasetoutflow.columns = self.datasetoutflow.loc["time_attr"]
        self.datasetoutflow.drop(["time_attr"], axis=0, inplace=True)
        time_range = self.datasetinflow.columns.value_counts().sort_index().index.tolist()
        time_map = dict(zip(time_range, list(range(len(time_range)))))
        self.data, self.inflow_matrices, self.outflow_matrices = _generate_data(self.datasetinflow, self.datasetoutflow, time_range=time_range,
                                   skills=self.datasetinflow.index.to_list(), class_num=class_num,
                                   min_length=min_length)
        graph = nx.read_gpickle(graph_dir)
        self.graph = from_networkx(graph, group_edge_attrs=['weight'])
        self.graph = convert_to_hetero(self.graph)
        self.max_length = max_length
        
        super().__init__(self.data, self.max_length, self.graph)

    def __len__(self):
        return len(self.data)
    
    def _get_graph(self):
        return self.graph

    
class SkillDataLoader(BaseDataLoader):
    """
    Skill data loader using BaseDataLoader
    """
    def __init__(self, data_dir, graph_dir, skill_num=11721, max_length=18, batch_size=1024, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = SkillflowDataset(data_dir, graph_dir, max_length=max_length, batch_size=batch_size)
        self.graphdata = self.dataset._get_graph()
        # self.data = DataLoader(self.dataset, batch_size, shuffle, pin_memory=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
#-----------------------------Utils-----------------------------

def convert_to_hetero(graph):
    data = HeteroData()
    data['node', 'is_parent', 'node'].edge_index = graph.edge_index[:, (graph.edge_attr.int() ==
                                                                        1).nonzero()[:,0]]
    data['node', 'is_child', 'node'].edge_index =graph.edge_index[:, (graph.edge_attr.int() ==
                                                                      2).nonzero()[:,0]]
    # data['node', 'temp', 'node'].edge_index =graph.edge_index[:, (graph.edge_attr.int() ==
    #                                                                  3).nonzero()[:,0]]
    data['node', 'relate', 'node'].edge_index =graph.edge_index[:, (graph.edge_attr.int() ==
                                                                     4).nonzero()[:,0]]
    return data