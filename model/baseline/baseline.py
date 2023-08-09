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

from.esg import ESG
from .wavenet import gwnet 
from .MTGNN.net import gtnet
from .Fedformer.models.FEDformer import Fedformer
from .Fedformer.models.Autoformer import Autoformer

class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [4]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 18
        label_len = 4
        pred_len = 1
        output_attention = False
        enc_in = 1
        dec_in = 1
        d_model = 32
        embed = 'timeF'
        dropout = 0.2
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 32
        e_layers = 2
        d_layers = 1
        c_out = 32
        activation = 'gelu'
        wavelet = 0
        
class BaseSeqModel(nn.Module):

    '''Basic sequential model for prediction, providing default operation, modules and loss functions
    '''

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def default_amplifier(self, embed_dim):
        return nn.Linear(1, embed_dim)

    def default_decoder(self, embed_dim, hidden_dim, dropout, class_num):
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, class_num)
        )

    def forward(self):
        raise NotImplementedError

    def _amplify(self, x, amplifier):
        x = x.unsqueeze(-1)
        y = amplifier(x)
        return y

    def _encode(self):
        raise NotImplementedError

    def _decode(self, x, decoder):
        y = decoder(x)
        y = F.log_softmax(y, dim=-1)
        return y

class Graph_Baseline(BaseSeqModel):
    def __init__(self, skill_num, embed_dim, skill_embed_dim, class_num, nhead, nlayers, dropout, model, *args, **kwargs):
        super().__init__()

        self.src_mask = None
        self.embed_dim = embed_dim
        self.skill_num = skill_num
        self.nlayer = nlayers
        self.model = model
        self.in_dim = embed_dim
        self.emb_comb = 2
        # dy_com_pos_hgnn
        self.demand_amplifier = self.default_amplifier(self.in_dim)
        self.supply_amplifier = self.default_amplifier(self.in_dim)
        self.seq_linear = nn.Sequential(
            nn.Linear(1, 4),
            nn.Linear(4, 4),
            )
        self.init_skill_emb1 = nn.Embedding(skill_num, skill_embed_dim)
        self.init_skill_emb2 = nn.Embedding(skill_num, skill_embed_dim)
        if self.model == "wavenet":
            self.stgnn = gwnet(num_nodes=skill_num,dropout=dropout, in_dim=self.in_dim, out_dim=1,supports=[1],residual_channels=self.in_dim,dilation_channels=self.in_dim,skip_channels=self.in_dim*4,end_channels=self.in_dim*8)
        if self.model == "mtgnn":
            self.stgnn = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=skill_num, device=1, static_feat=None, dropout=dropout, subgraph_size=skill_num, node_dim=embed_dim, dilation_exponential=3, conv_channels=8, residual_channels=8, skip_channels=8, end_channels=8, seq_length=18, in_dim=embed_dim, out_dim=embed_dim, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
        if self.model == "esg":
            self.stgnn = ESG(dy_embedding_dim=embed_dim,dy_interval=1, num_nodes=skill_num, seq_length=18, pred_len=1, in_dim=embed_dim, out_dim=embed_dim, n_blocks=2, n_layers=2, conv_channels=2, residual_channels=2, skip_channels=2, end_channels=2, kernel_set=[1], dilation_exp=1, gcn_depth=2, device=2, fc_dim=skill_embed_dim, st_embedding_dim=skill_embed_dim, static_feat=self.init_skill_emb1.weight)
        if self.model == "Fedformer":
            self.in_dim = 1
            self.demand_amplifier = self.default_amplifier(self.in_dim)
            self.supply_amplifier = self.default_amplifier(self.in_dim)
            configs = Configs()
            self.stgnn = Autoformer(configs)
            # self.stgnn = Fedformer(modes=embed_dim,seq_len=18,label_len=1,pred_len=1, enc_in=self.in_dim, dec_in=self.in_dim,moving_avg=[4],dropout=dropout, c_out=embed_dim)
            self.emb_comb=1
        self.init_start_token = nn.Embedding(1, skill_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead,dropout=dropout,batch_first=True)
        self.linear_for_trans = nn.Linear(6, embed_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.demand_MLP = nn.Sequential(
            nn.Linear(embed_dim* self.emb_comb, embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, class_num),
            )
        self.supply_MLP = nn.Sequential(
            nn.Linear(embed_dim* self.emb_comb, embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, class_num),
            ) 

    def forward(self, x, l, s, t_s, t_e, g, comm, skill_semantic_embed, gap):
        d_x, s_x = x  # [batch_size, 18]
        max_len = d_x.shape[1]
        d_x = d_x[:,:l]
        s_x = s_x[:,:l]
        if self.model != "Fedformer":
            d_x = F.pad(input=d_x, pad=(max_len - l, 0))
            s_x = F.pad(input=s_x, pad=(max_len - l, 0))
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
            if self.model == "wavenet":
                demand = F.relu(self.linear_for_trans(demand))
                supply = F.relu(self.linear_for_trans(supply))
            # demand = demand
            if self.model == "mtgnn":
                demand = demand.transpose(0,1)
                supply = supply.transpose(0,1)
            if self.model == "esg":
                demand = demand.transpose(0,1)
                supply = supply.transpose(0,1)
            d_y, s_y = self._decode((demand, supply), node_emb1) #[batch_size, 10]
        if self.model == "Fedformer":
            d_x = F.pad(input=d_x, pad=(max_len - l, 0))
            s_x = F.pad(input=s_x, pad=(max_len - l, 0))
            time_feature = (torch.arange(t_s[0], t_e[0]+1)/11.0-0.5).to(d_x.device).expand(d_x.shape[0],-1)
            label = time_feature[:,-1].unsqueeze(-1).unsqueeze(-1) + 1/11
            time_feature = F.pad(input=time_feature, pad=(max_len - l, 0)).unsqueeze(-1)
            # print(time_feature.shape)
            time_feature = self.seq_linear(time_feature)
            label = self.seq_linear(label)
            d_x = d_x.unsqueeze(-1)
            s_x = s_x.unsqueeze(-1)
            # d_x = self._amplify(d_x, self.demand_amplifier)
            # s_x = self._amplify(s_x, self.supply_amplifier)
            demand = self.stgnn(d_x, time_feature, label)
            supply = self.stgnn(s_x, time_feature, label)
            demand = demand[:,-1, :]
            supply = supply[:,-1, :]
            d_y = F.log_softmax(self.demand_MLP(demand),dim=-1)
            s_y = F.log_softmax(self.supply_MLP(supply),dim=-1)
            return (d_y, s_y), 0, 0, 0
            
        return (d_y, s_y), learned_graph, loss, pred
    def _decode(self, x, s):
        d_x, s_x = x

        # d_s = self.drop(self.inoutflow_merge(torch.concat([d_x, s_x, s], dim=-1)))
        d_y = self.demand_MLP(torch.concat([d_x, s[:,:]], dim=-1))
        s_y = self.supply_MLP(torch.concat([s_x, s[:,:]], dim=-1))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)