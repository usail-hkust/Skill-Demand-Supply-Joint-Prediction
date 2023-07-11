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
from hypnettorch.mnets.mlp import MLP
from hypnettorch.hnets.mlp_hnet import HMLP
from .wavenet import gwnet 

import copy
from typing import Optional, Tuple

from itertools import combinations

from .hyperlstm.hyperlstm import HyperLSTM

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


class Skill_Evolve_Hetero(BaseSeqModel):
    ''' Skill Graph Learning
    '''
    def __init__(self, skill_num, embed_dim, skill_embed_dim, class_num, nhead, nhid, nlayers, dropout, sample_node_num, model, hyperdecode=False, hypcomb='', *args, **kwargs):
        super().__init__()

        self.src_mask = None
        self.embed_dim = embed_dim
        self.nlayer = nlayers
        self.model = model
        self.hyper_size = int(embed_dim/4)
        self.hyperdecode = hyperdecode
        self.hyper_n_layers = 1

        # dy_com_pos_hgnn
        self.largedyskillhgnn = Crossview_Graph_Learning(skill_num, skill_embed_dim, sample_node_num)
        
        # amplifier
        self.inflow_amplifier = self.default_amplifier(embed_dim)
        self.outflow_amplifier = self.default_amplifier(embed_dim)
        
        self.inflow_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True)
        self.outflow_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True)
        
        self.init_skill_emb = nn.Embedding(skill_num, skill_embed_dim)
        
        
        # decoder
        self.inoutflow_merge = nn.Linear(embed_dim * 2+skill_embed_dim, embed_dim)
        self.before_hyp_MLP = nn.Sequential(nn.Linear(embed_dim*2, embed_dim))
        self.demand_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, class_num)
            )
        self.supply_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, class_num)
            ) 

    def forward(self, x, l, s, t_s, t_e, g, comm, skill_semantic_embed, gap):
        # print(gap.shape)
        '''
        Graph Evolve
        1. input
        '''
        d_x, s_x = x  # [batch_size, 18]

        l  = l.expand(d_x.shape[0])
        amp_d_x = self._amplify(d_x, self.inflow_amplifier) #[batch_size, 18, 128]
        amp_s_x = self._amplify(s_x, self.outflow_amplifier)

        enc_d_x_seq = self._encode(amp_d_x, l, self.init_skill_emb.weight, self.position_encoder, self.inflow_LSTM_encoder) #[batch_size, l, 128]
        enc_s_x_seq = self._encode(amp_s_x, l, self.init_skill_emb.weight, self.position_encoder, self.outflow_LSTM_encoder)

        
        embedding = self.init_skill_emb.weight
        org_node_emb, embedding, learned_graph, loss = self.largedyskillhgnn(enc_d_x_seq, enc_s_x_seq, l, t_s, t_e, g, comm, skill_semantic_embed, self.init_skill_emb.weight)

        enc_d_x = enc_d_x_seq[:,-1,:]
        enc_s_x = enc_s_x_seq[:,-1,:]
        d_y, s_y = self._decode((enc_d_x, enc_s_x), embedding) #[batch_size, 10]
        
        loss = 0
        learned_graph = 0
        pred = 0
        
        return (d_y, s_y), learned_graph, loss, pred
    
    def _amplify(self, x, amplifier_1):
        x = x.unsqueeze(-1)
        y = F.leaky_relu(amplifier_1(x))
        return y

    def _encode(self, x, l, init_emb, position_encoder, encoder, has_mask=True):
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x, (init_emb.expand(self.nlayer, -1, -1), torch.zeros(self.nlayer, init_emb.shape[0], init_emb.shape[1]).to(init_emb.get_device()))) # 
        # y, _ = pad_packed_sequence(y, batch_first=True)
        return y
    
    def _decode(self, x, s):
        d_x, s_x = x
        # d_s = self.inoutflow_merge(torch.concat([d_x, s_x, s], dim=-1))
        d_y = self.demand_MLP(self.before_hyp_MLP(torch.concat([d_x, s[:d_x.shape[0],:]], dim=-1)))
        s_y = self.supply_MLP(self.before_hyp_MLP(torch.concat([s_x, s[d_x.shape[0]:, :]], dim=-1)))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)

class AdaptiveGraph(nn.Module):
    '''Adaptive Graph Graph Neural Network
    '''
    def __init__(self, skill_num, skill_embed_dim, sample_node_num):
        super().__init__()
        self.sample_node_num = sample_node_num
        self.skill_num = skill_num
        self.GNN_0 = GCN(skill_embed_dim)
        self.GNN_1 = GCN(skill_embed_dim)
        self.skill_emb_1 = nn.Embedding(skill_num, skill_embed_dim)
        self.skill_emb_2 = nn.Embedding(skill_num, skill_embed_dim)

        self.fuse_seq = nn.Linear(skill_embed_dim*3, skill_embed_dim)
        self.fuse_E1_E2 = nn.Linear(skill_embed_dim*2, skill_embed_dim)
        self.fuse_seq_node = nn.Linear(skill_embed_dim*2, skill_embed_dim)
        self.loss_mse = nn.MSELoss()
        self.L = torch.nn.L1Loss()

    def forward(self, inflow_seq_emb, outflow_seq_emb, l, t_s, t_e, g_d, comm, skill_semantic_embed,init_emb):
        '''
        :param cat_emb: [N, seq_len, dim] g: [adj list]
        :return: [N, dim]
        '''
        loss = 0
        seq_cat = torch.cat([inflow_seq_emb, outflow_seq_emb],dim=-1)
        
        update_graph_emb = self.fuse_seq(torch.cat([init_emb, seq_cat[:, -1, :]],dim=-1))
       
        pred_g, update_list, edge_attr = self.graph_update(update_graph_emb, update_graph_emb)
        skill_embs = self.GNN_0(init_emb, update_list, edge_attr)
        skill_embs_2 = self.GNN_1(init_emb, g_d.edge_index[:,(g_d.edge_index<self.skill_num).all(0)], g_d.edge_attr[(g_d.edge_index<self.skill_num).all(0)])
        
        skill_embs = skill_embs + skill_embs_2
        
        return self.skill_emb_1 , skill_embs, pred_g, loss

    
    def graph_update(self, s_emb_1,s_emb_2):
        ''' Hetero graph_sampling
        '''
        # print(g.shape)
        # gt_adj = geo_utils.to_dense_adj(g, max_num_nodes=s_emb.shape[0])
        updated_adj = self.find_possible_edges('{i}', s_emb_1, s_emb_2)
        # print(updated_adj.shape)
        updated_adj = torch.where(updated_adj>0.2, updated_adj, 0)
        # print(updated_adj.shape)
        update_list, edge_attr = geo_utils.dense_to_sparse(updated_adj)
        return updated_adj, update_list, edge_attr
    def find_possible_edges(self, edge_type, embeddings, embeddings_2):
        n, dim = embeddings.shape    
        adj = F.softmax(torch.matmul(embeddings, embeddings_2.transpose(0,1)), dim=1)
        return adj
    
class Crossview_Graph_Learning(nn.Module):
    '''Dynamic Skill Heterogeneous Graph Neural Network
    '''
    def __init__(self, skill_num, skill_embed_dim, sample_node_num, model):
        super().__init__()
        self.sample_node_num = sample_node_num
        self.disentangle_num = 0
        self.model = model
        self.lamb = 0
        self.alpha = 128
        self.cluster_size = 100
        self.skill_num = skill_num
        self.GNN_0 = GCN(skill_embed_dim)
        self.GNN_1 = GCN(skill_embed_dim)
        if self.model == "semantic":
            self.GNN_2 = GCN(skill_embed_dim)
        if self.model == "hier":
            self.GNN_pool = GCN(skill_embed_dim)
        self.skill_emb_1 = nn.Embedding(skill_num, skill_embed_dim)
        # self.skill_emb_2 = nn.Embedding(skill_num, skill_embed_dim)
        self.skill_inflow_LT = nn.Linear(skill_embed_dim, skill_embed_dim, bias=True)
        self.skill_outflow_LT = nn.Linear(skill_embed_dim, skill_embed_dim, bias=True)
        self.multihead_attn = nn.MultiheadAttention(skill_embed_dim, 4,dropout=0.2,batch_first=True)
        # self.skill_inflow_emb = nn.Embedding(skill_num, skill_embed_dim)
        # self.densecat_linear = nn.Linear(skill_embed_dim*2, skill_embed_dim)
        if model == "semantic":
            self.Transform_semantic_embed = nn.Linear(1536, skill_embed_dim)
            self.distributed_embed = nn.Linear(skill_embed_dim*3, 3)

        
        self.pool_mapping = nn.Linear(skill_embed_dim, self.cluster_size)
        self.fuse_edge_seq =nn.Sequential(
            nn.Linear(skill_embed_dim*2, skill_embed_dim)
            )
        self.fuse_emb = nn.Linear(skill_embed_dim*2, skill_embed_dim)
        self.fuse_pool = nn.Linear(skill_embed_dim*2, 1)

        self.loss_mse = nn.MSELoss()
        self.L = torch.nn.L1Loss()

    def forward(self, inflow_seq_emb, outflow_seq_emb, l, t_s, t_e, g_d, comm, skill_semantic_embed, init_emb):
        '''
        :param cat_emb: [N, seq_len, dim] g: [adj list]
        :return: [N, dim]
        '''
        loss = 0
        inflow_emb_1 = self.skill_inflow_LT(init_emb)
        outflow_emb_1 = self.skill_outflow_LT(init_emb)

        cat = torch.cat([inflow_emb_1,outflow_emb_1], dim=0)
        
        seq_cat = torch.cat([inflow_seq_emb, outflow_seq_emb],dim=0)
        
        # seq_cat = torch.max(seq_cat, dim=1)[0]
        ### HyperLSTM
        seq_cat = seq_cat[:, -1, :]

        
        # graph_emb = torch.cat([self.graph_emb.weight, seq_cat],-1)
        update_graph_edge_emb = self.fuse_edge_seq(torch.cat([cat, seq_cat], dim=-1))
        # update_graph_edge_emb_2 = self.fuse_edge_seq(torch.cat([cat_2, seq_cat], -1))
        # graph_emb = self.fuse_emb(torch.cat([graph_emb, seq_cat], -1))
        pred_g, update_list, edge_attr = self.graph_update(update_graph_edge_emb,update_graph_edge_emb)
        # skill_embs = self.GNN(cat, update_list, edge_attr)
        skill_embs_ada = self.GNN_0(update_graph_edge_emb, update_list, edge_attr)
        skill_embs_cooc = self.GNN_1(update_graph_edge_emb, g_d.edge_index, g_d.edge_attr)
        if self.model == "semantic":
            semantic_view = self.Transform_semantic_embed(skill_semantic_embed.weight)
            skill_embs_sem = self.GNN_2(semantic_view, update_list[:,(update_list<self.skill_num).all(0)], edge_attr[(update_list<self.skill_num).all(0)])
            skill_embs_sem = torch.cat([skill_embs_sem, skill_embs_sem], dim=0)
            skill_embs = skill_embs_ada+skill_embs_cooc+skill_embs_sem
        
        skill_embs = skill_embs_ada+skill_embs_cooc

        if self.model == "hier":
            skill_hier_embs, loss = self.heirarchical(skill_embs, g_d)
            skill_embs = skill_embs + skill_hier_embs
        return cat, skill_embs, pred_g, loss

        return cat, combined_multi_hop, total_loss # [skill_num, embedding_dim]
    
    def heirarchical(self, embs, adj):
        s = self.pool_mapping(embs)
        s = torch.softmax(s, dim=-1)
        adj = geo_utils.to_dense_adj(edge_index=adj.edge_index, edge_attr=adj.edge_attr)
        adj = adj.squeeze()
        out, out_adj, link_loss, ent_loss = dense_adaptive_diff_pool(embs, adj, s)
        out = embs[torch.argmax(s, dim=0), :]
        # out = embs[:self.cluster_size, :]
        update_list, edge_attr = geo_utils.dense_to_sparse(out_adj)
        pooled = self.GNN_pool(out, update_list, edge_attr)
        pooled = pooled.squeeze()
        # mapped_index = torch.argmax(s,dim=1)
        # print(mapped_index.shape)
        # print(s.shape)
        # print(pooled.shape)
        update_to_org, weight = self.attention(embs,pooled,pooled)
        # update_to_org = torch.matmul(s, pooled)
        # atten = torch.matmul([embs, update_to_org],dim=-1)
        # atten = F.sigmoid(self.fuse_pool(atten))
        pooled_embs = embs + update_to_org


        # pooled_mebs = embs + atten * s @ pooled
        return pooled_embs, link_loss + ent_loss
    
    def graph_update(self, s_emb, s_emb_2):
        updated_adj = self.find_possible_edges('{i}', s_emb, s_emb_2)
        # threshold = updated_adj.sum()+updated_adj.std()
        # # larger than threshold
        # updated_adj = torch.where(updated_adj>threshold, updated_adj, 0)
        # # topk
        # mask = torch.zeros(updated_adj.size(0), updated_adj.size(0)).to(updated_adj.device)
        # mask.fill_(float('0'))
        # s1,t1 = updated_adj.topk(int(updated_adj.size(0)*0.1),1)
        # mask.scatter_(1,t1,s1.fill_(1))
        # updated_adj = updated_adj*mask

        # update_list, edge_attr = geo_utils.dense_to_sparse(updated_adj)
        updated_adj = torch.where(updated_adj>0.2, updated_adj, 0)
        update_list, edge_attr = geo_utils.dense_to_sparse(updated_adj)
        return updated_adj, update_list, edge_attr
    
    def find_possible_edges(self, edge_type, embeddings, embeddings_2):
        n, dim = embeddings.shape
        adj = F.softmax(F.relu(torch.matmul(embeddings, embeddings_2.transpose(0,1))), dim=1)
        return adj
        dy_sent = torch.unsqueeze(torch.relu(embeddings), dim=1).repeat(1, n, 1)
        dy_revi = dy_sent.transpose(0,1)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        # print(self.conv[edge_type])
        support = self.mlp1[edge_type](y, (n, n))
        mask = torch.sigmoid(self.mlp2[edge_type](y,(n, n)))
        support = support * mask
        sub_adj = (1-mask) * sub_adj + mask * support
        sub_adj = F.relu(sub_adj)
        return sub_adj
    def find_possible_edges_ver2(self, edge_type, embedding, sub_adj):
        return


class GCN(nn.Module):
    '''SKill Heterogeneous Graph Neural Network
    '''

    def __init__(self, skill_embed_dim: int):
        super().__init__()
        self.preserve_ratio = 0.1
        # self.GCN_1 = DCRNN(skill_embed_dim, skill_embed_dim, K=5)
        self.GCN_1 = geo_nn.conv.GCNConv(skill_embed_dim, skill_embed_dim, dropout=0.1, normalize=True)
        self.GCN_2 = geo_nn.conv.GCNConv(skill_embed_dim, skill_embed_dim, dropout=0.1, normalize=True)
        # self.GCN_1 = geo_nn.conv.SAGEConv(skill_embed_dim, skill_embed_dim, dropout=0.2)
        # self.GCN_2 = geo_nn.conv.SAGEConv(skill_embed_dim, skill_embed_dim, dropout=0.2)
    def forward(self, skill_embed, adj_list, edge_attr):
        temp = skill_embed
        out = self.GCN_1(skill_embed, adj_list, edge_attr)
        temp = (1-self.preserve_ratio)* out+self.preserve_ratio *temp
        out = self.GCN_2(temp, adj_list, edge_attr)
        temp = (1-self.preserve_ratio)* out+self.preserve_ratio *temp
        # temp = self.GCN_1(skill_embed, adj_list)
        # temp = self.GCN_2(temp, adj_list)
        # temp = self.heterognn_2(temp,adj)
        # temp = self.heterognn_2(temp, adj)
        return temp
    

def dense_adaptive_diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    '''Adaptive pooling with specified clustering number
    '''
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask
    # print(s.shape)
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
    # clus_loss = 
    # temp = self.heterognn_2(temp,adj)
    # temp = self.heterognn_2(temp, adj)
    return out, out_adj, link_loss, ent_loss