import torch
import pandas as pd
import os 
import networkx as nx
import pickle
from community import community_louvain
import pickle
# import community.community_louvain as community_louvain
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch_geometric.utils import to_networkx, from_networkx, subgraph
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
        g = sample['G']
        t_s = sample['Sta']
        t_e = sample['End']
        
        l = sample['L']
        s = sample['S']
        # padding
        d_x_padded = F.pad(input=d_x, pad=(0, self.max_length - l))
        s_x_padded = F.pad(input=s_x, pad=(0, self.max_length - l))
        # d_x_padded = d_x
        # s_x_padded = s_x
        return (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e, g

class SkillflowDataset(SingleSkillDataset):
#     data dir - ./data/skill_inflow_outflow
#     graph_dir = ./data/myGraph.gpickle
    
    def __init__(self, data_dir, graph_dir, batch_size, max_length, shuffle=True, validation_split=0.0,skill_num=16956,subgraph_num=16956, class_num=10, min_length=5, num_workers=1, training=True, *args, **kwargs):
        self.data_dir = data_dir
        subgraph_num=7446
        self.subgraph_num = subgraph_num
        startdate = pd.Timestamp(2017,10,1)
        enddate = pd.Timestamp(2019,4,1)
        period = 'M'
        time_attr = pd.date_range(start=startdate, end=enddate, freq=period).to_period(period)
        time_attr = time_attr.to_series().astype(str)
        dataset = kwargs.get("dataset")
        self.datasetdemand =  pd.read_csv(
            os.path.join(data_dir, f"skill_inflow_list_{dataset}.csv"),
            encoding='utf-8',
            header=0,
            on_bad_lines='warn',
        ).T
        
        self.datasetdemand.columns = time_attr
        self.datasetdemand.drop("time_attr", axis=0, inplace=True, errors="ignore")
        self.datasetsupply =  pd.read_csv(
            os.path.join(data_dir, f"skill_outflow_list_{dataset}.csv"),
            encoding='utf-8',
            header=0,
            on_bad_lines='warn'
        ).T
        self.classification = torch.from_numpy(pd.read_csv(
            os.path.join(data_dir, f"classification.csv"),
            encoding='utf-8',
            header=0,
            on_bad_lines='warn'
        )[:subgraph_num]["type"].values)
        self.datasetsupply.columns = time_attr
        self.datasetsupply.drop(["time_attr"], axis=0, inplace=True, errors="ignore")
        # Set subgraph
        self.datasetdemand = self.datasetdemand[:subgraph_num]
        self.datasetsupply = self.datasetsupply[:subgraph_num]


        time_range = self.datasetdemand.columns.value_counts().sort_index().index.tolist()
        time_map = dict(zip(time_range, list(range(len(time_range)))))
        self.data, self.demand_matrices, self.supply_matrices = _generate_data(self.datasetdemand, self.datasetsupply, time_range=time_range,
                                   skills=self.datasetdemand.index.to_list(), class_num=class_num,
                                   min_length=min_length)
        self.demand_graph = []

        ### Demand_graph_w_adj_matri
        edge_index = torch.load(graph_dir+'edge_index.pt')
        edge_attr = torch.load(graph_dir+'edge_attr.pt')
        x = torch.zeros(subgraph_num*2,1)
        edge_index= edge_index[:, (edge_attr>=5e-4).squeeze()]
        edge_attr= edge_attr[(edge_attr>=5e-4).squeeze()]
        if subgraph_num!=skill_num:
            subset = [i for i in range(subgraph_num)]+[i+skill_num for i in range(subgraph_num)]
            x = torch.zeros(subgraph_num*2,1)
            edge_index, edge_attr = subgraph(subset,edge_index,edge_attr,relabel_nodes=True,num_nodes=skill_num*2)
        self.demand_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(edge_attr[(edge_index<subgraph_num).all(0)].shape)
        print(edge_attr[(edge_index>=subgraph_num).all(0)].shape)
        weight = torch.FloatTensor(np.loadtxt("./data/embedding.txt"))
        self.skill_semantic_emb = torch.nn.Embedding.from_pretrained(weight, freeze=True)
        # G2 = graph.to_undirected()
        # self.graph = convert_to_stack_data(self.graph, self.disentangle)
        # with open("/home/wchao/SkillTrendPrediction/data/skill_community", "rb") as fp:
        #     l = pickle.load(fp)
        # # comm =  community_louvain.best_partition(G2)
        # # # valid_data_loader.dataset.comm
        # # l = []
        # # for k in comm:
        # #     while comm[k]>=len(l):
        # #         l.append([])
        # #     l[comm[k]].append(k)
        #     self.comm = l
    
        # self.graph = convert_to_hetero(self.graph)
        self.max_length = max_length
        
        super().__init__(self.data, self.max_length, self.demand_graph)

    def __len__(self):
        return len(self.data)
    
    def _get_graph(self):
        return self.demand_graph
    def _get_emb(self):
        return self.skill_semantic_emb

    
class SkillDataLoader(BaseDataLoader):
    """
    Skill data loader using BaseDataLoader
    """
    def __init__(self, data_dir, graph_dir, skill_num=7446, max_length=18, batch_size=1024, shuffle=True, validation_split=0.0, class_num=10, min_length=5, num_workers=1, training=True, *args, **kwargs):
        self.data_dir = data_dir
        self.dataset = SkillflowDataset(data_dir, graph_dir, max_length=max_length, batch_size=batch_size, class_num=class_num, min_length=min_length, *args, **kwargs)
        self.graphdata = self.dataset._get_graph()
        self.skill_semantic_emb = self.dataset._get_emb()
        # self.data = DataLoader(self.dataset, batch_size, shuffle, pin_memory=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
#-----------------------------Utils-----------------------------

def convert_to_hetero(graph):
    data = HeteroData()
    data['node', 'is_parent', 'node'].edge_index = graph.edge_index[:, (graph.edge_attr.int() ==
                                                                        2).nonzero()[:,0]]
    data['node', 'is_child', 'node'].edge_index =graph.edge_index[:, (graph.edge_attr.int() ==
                                                                      3).nonzero()[:,0]]
    # data['node', 'temp', 'node'].edge_index =graph.edge_index[:, (graph.edge_attr.int() ==
    #                                                                  3).nonzero()[:,0]]
    data['node', 'relate', 'node'].edge_index = graph.edge_index[:, (graph.edge_attr.int() ==
                                                                     4).nonzero()[:,0]]
    return data

def convert_to_stack_data(graph, disentangle):
    datalist = []
    for i in range(disentangle):
        data = Data()
        data.edge_index = graph.edge_index[:, (graph.edge_attr.int() ==
                                            i).nonzero()[:,0]]
        datalist.append(data)
    return datalist

def convert_to_dense(graph):
    data = Data(edge_index=graph.edge_index)
    return data