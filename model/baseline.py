import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import networkx as nx
import math
from base import BaseModel
import torch_geometric.nn as geo_nn
import torch_geometric.utils as geo_utils
from .model import BaseSeqModel
import copy
from typing import Optional, Tuple

from itertools import combinations


class BaseRNNModel(BaseSeqModel):
    '''Override BaseSeqModel for recurrent neural networks variants, which can easily used based on this class
    '''
    # "type": "RNN",
    #         "args":{
    #             "embed_dim": 128,
    #             "class_num": 5,
    #             "hidden_dim": 32,
    #             "rnn_layer_num": 3,
    #             "dropout": 0.2
    #         }
    def __init__(
        self,
        ENCODER: nn.Module,
        embed_dim: int,
        rnn_layer_num: int,
        class_num: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()
        
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.demand_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.supply_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.demand_decoder = self.default_decoder(embed_dim, embed_dim, dropout, class_num)
        self.supply_decoder = self.default_decoder(embed_dim, embed_dim, dropout, class_num)

    def forward(self, x, l, s, t_s, t_e, g, g_1,g_2, gap):
        d_x, s_x = x
        # print(s_x.shape)
        l  = l.expand(d_x.shape[0])
        # print(d_x.shape)
        d_x = self._amplify(d_x, self.demand_amplifier)
        s_x = self._amplify(s_x, self.supply_amplifier)
        # print(d_x.shape)
        d_x = self._encode(d_x, l, self.demand_encoder)
        s_x = self._encode(s_x, l, self.supply_encoder)
        d_y = self._decode(d_x, self.demand_decoder)
        s_y = self._decode(s_x, self.supply_decoder)
        return (d_y, s_y) , 0, 0, 0

    def _encode(self, x, l, encoder):
        
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x)
        y, _ = pad_packed_sequence(y, batch_first=True)
        # y = _get_masked_seq_last(y, l)
        y = y[:, -1,:]
        return y
    
class RNN(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.RNN, *args, **kwargs)


class GRU(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.GRU, *args, **kwargs)


class LSTM(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.LSTM, *args, **kwargs)

class Transformer(BaseSeqModel):
    # "type": "RNN",
    #         "args":{
    #             "embed_dim": 128,
    #             "class_num": 5,
    #             "hidden_dim": 32,
    #             "rnn_layer_num": 3,
    #             "dropout": 0.2
    #         }
    def __init__(
        self,
        ENCODER: nn.Module,
        embed_dim: int,
        rnn_layer_num: int,
        class_num: int,
        hidden_dim: int,
        nhead: int,
        nhid: int, 
        nlayers: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.demand_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, embed_dim, dropout), nlayers)
        self.supply_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, embed_dim, dropout), nlayers)
        self.position_encoder = PositionalEncoding(embed_dim, dropout)

        self.demand_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)
        self.supply_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)

    def forward(self, x, l, s, t_s, t_e, g, g_1,g_2):
        d_x, s_x = x
        # print(s_x.shape)
        l  = l.expand(d_x.shape[0])
        # print(d_x.shape)
        d_x = self._amplify(d_x, self.demand_amplifier)
        s_x = self._amplify(s_x, self.supply_amplifier)
        # print(d_x.shape)
        d_x = self._encode(d_x, l, self.position_encoder, self.demand_encoder)
        s_x = self._encode(s_x, l, self.position_encoder, self.supply_encoder)
        d_y = self._decode(d_x, self.demand_decoder)
        s_y = self._decode(s_x, self.supply_decoder)
        
        return (d_y, s_y) , 0, 0, 0

    def _encode(self, x, l, position_encoder, transformer_encoder, has_mask=True):
        x = x.permute(1, 0, 2)
        # mask
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = _generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        x = x * math.sqrt(self.embed_dim)
        x = position_encoder(x)
        y = transformer_encoder(x, self.src_mask)
        y = y.permute(1, 0, 2)

        return y
class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels,
                                   hidden_channels))
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class Graph_Baseline(BaseSeqModel):
    def __init__(self, skill_num, embed_dim, skill_embed_dim, class_num, nhead, nlayers, dropout, model, *args, **kwargs):
        super().__init__()

        self.src_mask = None
        self.embed_dim = embed_dim
        self.skill_num = skill_num
        self.nlayer = nlayers
        self.model = model
        self.in_dim = 32
        # dy_com_pos_hgnn
        self.demand_amplifier = self.default_amplifier(self.in_dim)
        self.supply_amplifier = self.default_amplifier(self.in_dim)
        self.init_skill_emb1 = nn.Embedding(skill_num, skill_embed_dim)
        self.init_skill_emb2 = nn.Embedding(skill_num, skill_embed_dim)
        if self.model == "wavenet":
            self.stgnn = gwnet(num_nodes=skill_num,in_dim=self.in_dim, out_dim=1,supports=[1],residual_channels=self.in_dim,dilation_channels=self.in_dim,skip_channels=self.in_dim*4,end_channels=self.in_dim*8)
        if self.model == "mtgnn":
            self.stgnn = gwnet()
        
        self.init_start_token = nn.Embedding(1, skill_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead,dropout=dropout,batch_first=True)
        self.linear_for_trans = nn.Linear(6, embed_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.demand_MLP_1 = nn.Linear(embed_dim+skill_embed_dim, embed_dim)
        self.supply_MLP_1 = nn.Linear(embed_dim+skill_embed_dim, embed_dim)
        self.inflow_decoder = nn.Linear(embed_dim, class_num)
        self.outflow_decoder = nn.Linear(embed_dim, class_num)
        self.demand_MLP_2 = nn.Linear(embed_dim, embed_dim)
        self.supply_MLP_2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, l, s, t_s, t_e, g, comm, skill_semantic_embed, gap):
        d_x, s_x = x  # [batch_size, 18]
        # s_subgraph = torch.range(self.skill_num,self.skill_num*2, dtype=torch.long)
        g_d_idx, g_d_attr = g.edge_index[:,(g.edge_index<self.skill_num).all(0)], g.edge_attr[(g.edge_index<self.skill_num).all(0)]
        g_s_idx, g_s_attr = g.edge_index[:,(g.edge_index>=self.skill_num).all(0)]-self.skill_num, g.edge_attr[(g.edge_index>=self.skill_num).all(0)]
        # g_s_idx, g_s_attr = geo_utils.subgraph(s_subgraph,g.edge_index,g.edge_attr,relabel_nodes=True)
        g_d = geo_utils.to_dense_adj(g_d_idx, edge_attr= g_d_attr,max_num_nodes=self.skill_num).squeeze()
        g_s = geo_utils.to_dense_adj(g_s_idx, edge_attr= g_s_attr,max_num_nodes=self.skill_num).squeeze()
        # print(g_s.shape)
        # d_x = d_x.squeeze()
        # s_x = s_x.squeeze()
        learned_graph, loss, pred=0, 0, 0
        d_x = self._amplify(d_x, self.demand_amplifier).unsqueeze(0)
        s_x = self._amplify(s_x, self.supply_amplifier).unsqueeze(0)
        d_x = d_x.permute((0,3,1,2))  # [1, in_dim, num_skill, seq_length]
        s_x = s_x.permute((0,3,1,2))  # [1, in_dim, num_skill, seq_length]
        
        node_emb1 = self.init_skill_emb1.weight
        node_emb2 = self.init_skill_emb2.weight



        demand = self.stgnn(d_x,g_d,node_emb1,node_emb2)
        supply = self.stgnn(s_x,g_s,node_emb1,node_emb2) # [1, out_dim, num_skill, seq_length/3]
        # print(demand.shape)
        demand = demand.squeeze()
        supply = supply.squeeze()
        demand = F.relu(self.linear_for_trans(demand))
        supply = F.relu(self.linear_for_trans(supply))
        # demand = demand

        
        # print(supply.shape)
        d_y, s_y = self._decode((demand, supply), node_emb1) #[batch_size, 10]
        

        
        return (d_y, s_y), learned_graph, loss, pred
    def _decode(self, x, s):
        d_x, s_x = x

        # d_s = self.inoutflow_merge(torch.concat([d_x, s_x, s], dim=-1))
        d_y = self.inflow_decoder(F.leaky_relu(self.demand_MLP_2(self.drop(F.leaky_relu(self.demand_MLP_1(torch.concat([d_x, s[:,:]], dim=-1)))))))
        s_y = self.outflow_decoder(F.leaky_relu(self.supply_MLP_2(self.drop(F.leaky_relu(self.supply_MLP_1(torch.concat([s_x, s[:, :]], dim=-1)))))))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
# ----------------------Evolve Graph Models----------------------