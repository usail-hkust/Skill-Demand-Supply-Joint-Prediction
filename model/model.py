import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from base import BaseModel
import torch_geometric.nn as geo_nn
import torch_geometric.utils as geo_utils
import copy
# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

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

# ----------------------RNNs----------------------

class BaseRNNModel(BaseSeqModel):
    '''Override BaseSeqModel for recurrent neural networks variants, which can easily used based on this class
    '''
    # "type": "RNN",
            # "args":{
            #     "embed_dim": 128,
            #     "class_num": 10,
            #     "hidden_dim": 64,
            #     "rnn_layer_num": 3,
            #     "dropout": 0.2
            # }
    def __init__(
        self,
        hgs,
        ENCODER: nn.Module,
        embed_dim: int,
        rnn_layer_num: int,
        class_num: int,
        hidden_dim: int,
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
        self.demand_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)
        self.supply_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)

    def forward(self, x, l, s, t_s, t_e):
        d_x, s_x = x
        # print(d_x.shape, l.shape)
        d_x = self._amplify(d_x, self.demand_amplifier)
        # print(d_x.shape)
        s_x = self._amplify(s_x, self.supply_amplifier)
        d_x = self._encode(d_x, l, self.demand_encoder)
        s_x = self._encode(s_x, l, self.supply_encoder)
        d_y = self._decode(d_x, self.demand_decoder)
        s_y = self._decode(s_x, self.supply_decoder)
        return (d_y, s_y) , 0

    def _encode(self, x, l, encoder):
        
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x)
        y, _ = pad_packed_sequence(y, batch_first=True)
        y = _get_masked_seq_last(y, l)
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
    
    
# ----------------------Evolve Graph Models----------------------


class Skill_Evolve_Hetero(BaseSeqModel):
    '''Dynamic Heterogeneous Graph Enhanced Meta-learner
    '''
    # "type": "Skill_Evolve_Hetero",
    # "args":{
    #      "skill_num": 11721,
    #      "embed_dim": 128,
    #      "skill_embed_dim": 128,
    #      "class_num": 10,
    #      "hidden_dim": 64,
    #     "nhead": 4,
    #      "nhid": 16,
    #      "nlayers": 2,
    #      "dropout": 0.2
    #       }
    def __init__(self, hgs, skill_num, embed_dim, skill_embed_dim, class_num, hidden_dim, nhead, nhid, nlayers, dropout, *args, **kwargs):
        super().__init__()

        self.src_mask = None
        self.embed_dim = embed_dim

        # dy_com_pos_hgnn
        self.current_batch = 0
        self.dyskillhgnn = DySkillHGNN(hgs, skill_num, skill_embed_dim)
        
        # self.init_skill_emb = self.dyskillhgnn.init_skill_emb
        # amplifier
        self.inflow_amplifier = self.default_amplifier(embed_dim)
        self.outflow_amplifier = self.default_amplifier(embed_dim)
        self.skill_amplifier = nn.Linear(embed_dim + skill_embed_dim, embed_dim)
        # encoder
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout), nlayers)
        self.inflow_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True)
        self.outflow_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True)
        # self.demand_position_encoder = self.position_encoder
        # self.supply_position_encoder = self.position_encoder
        # self.demand_transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout), nlayers)
        # self.supply_transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout), nlayers)
        # self.testlinear = nn.Linear(embed_dim*18+128, 10)
        
        # decoder

        self.inoutflow_merge = nn.Linear(embed_dim * 3, embed_dim)
        self.inflow_attention = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.outflow_attention = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.inflow_decoder = nn.Linear(embed_dim, class_num)
        self.outflow_decoder = nn.Linear(embed_dim, class_num)
        # self.loss = nn.PoissonNLLLoss()

    def forward(self, x, l, s, t_s, t_e):
        self.current_batch = self.current_batch+1
        # for name, W in self.named_parameters():
        #     print(name, W.device)

        d_x, s_x = x  # [batch_size, 18]
        # s_emb = self.dyskillhgnn(s, t_s, t_e)  # [batch_size, 128]
        s_emb = self.dyskillhgnn(s, t_s, t_e)  # [batch_size, 18, 128]
        # print(s_emb.shape)
        # print(d_x.shape)
        amp_d_x = self._combined_amplify(d_x, s_emb, self.inflow_amplifier, self.skill_amplifier) #[batch_size, 18, 128]
        amp_s_x = self._combined_amplify(s_x, s_emb, self.outflow_amplifier, self.skill_amplifier)
        
        
        enc_d_x = self._encode(amp_d_x, torch.add(t_s, l), self.position_encoder, self.inflow_LSTM_encoder ) #[batch_size, 18, 128]
        enc_s_x = self._encode(amp_s_x, torch.add(t_s, l), self.position_encoder, self.outflow_LSTM_encoder)
        # print(enc_d_x.shape)


        cat_emb = torch.cat([enc_d_x,enc_s_x], dim=-1)
        loss = 0
        # print(self.current_batch)
        # if self.current_batch < 12000: 
        #     loss = self.dyskillhgnn.subgraph_sampling(s, cat_emb, t_s, t_e, self.current_batch)

        # print(enc_d_x.shape)
        # print(t_e.unsqueeze(0).shape)
        # print(torch.squeeze(s_emb.gather(0,t_e.view(1,-1,1).expand(1,-1,s_emb.size(2)))))
        # print(s_emb.gather(0,t_e.view(1,-1,1).expand(1,-1,s_emb.size(2))).shape)
        s_emb = s_emb.transpose(0, 1)
        seq_len = t_e.view(-1,1,1).expand(s_emb.size(0), 1, s_emb.size(2))
        # print(torch.squeeze(s_emb.gather(1, seq_len)).shape)
        d_y, s_y = self._decode((enc_d_x, enc_s_x), torch.squeeze(s_emb.gather(1, seq_len))) #[batch_size, 10]
        return (d_y, s_y), loss

    def _combined_amplify(self, x, s, amplifier_1, amplifier_2):
        x = x.unsqueeze(-1)
        y = F.leaky_relu(amplifier_1(x))
        y = torch.concat([y, torch.transpose(s, 0, 1)], dim=-1)
        y = F.leaky_relu(amplifier_2(y))
        return y
    def _amplify(self, x, s, amplifier_1, amplifier_2):
        x = x.unsqueeze(-1)
        y = amplifier_1(x)
        y = torch.concat([y, _repeat_seq_dim(s, y.size(1))], dim=-1)
        y = amplifier_2(y)
        return y

    def _encode_Trans(self, x, l, position_encoder, transformer_encoder, has_mask=True):
        x = x.permute(1, 0, 2)
        # mask
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = _generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        # encode
        x = x * math.sqrt(self.embed_dim)
        x = position_encoder(x)
        y = transformer_encoder(x, self.src_mask)
        y = _get_masked_seq_last(y, l, batch_first=False)

        return y
    def _encode(self, x, l, position_encoder, encoder, has_mask=True):
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        # print(x.shape)
        y, _ = encoder(x)
        y, _ = pad_packed_sequence(y, batch_first=True)
        # print(y.shape)
        y = _get_masked_seq_last(y, l)
        return y
    def _combined_decode(self, x, s):
        d_x, s_x = x
        d_s = self.inoutflow_merge(torch.concat([d_x, s_x, s], dim=-1))
        d_y = self.inflow_decoder(self.inflow_attention(torch.concat([d_x, d_s], dim=-1)))
        s_y = self.outflow_decoder(self.outflow_attention(torch.concat([s_x, d_s], dim=-1)))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
    def _decode(self, x, s):
        d_x, s_x = x
        d_s = self.inoutflow_merge(torch.concat([d_x, s_x, s], dim=-1))
        d_y = self.inflow_decoder(F.leaky_relu(self.inflow_attention(torch.concat([d_x, d_s], dim=-1))))
        s_y = self.outflow_decoder(F.leaky_relu(self.outflow_attention(torch.concat([s_x, d_s], dim=-1))))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)


class StSkillHGNN(nn.Module):
    '''Static Skill Heterogeneous Graph Neural Network
    '''

    def __init__(self, hgs, skill_num, skill_embed_dim):
        super().__init__()
        self.n = len(hgs)
        self.hg = hgs[0]
        self.hgnn = HeteroHGNN(skill_embed_dim)
        # .to(list(self.hg.edge_index_dict.values())[0].device)
        self.init_skill_emb = nn.Embedding(skill_num, skill_embed_dim)
        self.skill_embs = []
        self.skill_emb = {"node": self.init_skill_emb.weight}
        # for i in hgs:
        # self.skill_emb = self.hgnn({"node": self.init_skill_emb.weight}, self.hg.edge_index_dict)
            # self.skill_embs.append(self.skill_emb["node"])
        # self.skill_embs = [skill_emb]
        # self.com_embs = torch.stack(self.com_embs).permute(1, 2, 0)  # [n_com, dim, n_time]
        # self.pos_embs = torch.stack(self.pos_embs).permute(1, 2, 0)  # [n_pos, dim, n_time]
        # self.attn = nn.Parameter(torch.ones(self.n))
        # print(self.skill_embs)
        # self.skill_embs = torch.stack(self.skill_embs) # [n_step, skill_num, embedding_dim]
        # self.skill_embs = self.init_skill_emb.weight.unsqueeze(0).expand(self.n,-1,-1)

    def forward(self, s, t_s, t_e):
        if self.skill_emb["node"].device is not s.device:
            self.skill_emb["node"] = self.skill_emb["node"].to(s.device)
        temp = self.hgnn(self.skill_emb, self.hg.edge_index_dict)

        # print(temp)
        # for name, W in self.named_parameters():
        #     print(name, W.device)
        # print(self.hgnn)
        # print(s.device)
        # print(t_e.device)
        # print(self.init_skill_emb.weight.data)
        # print(self.skill_embs.data)
        # skill_embs_output = torch.stack(self.skill_embs)
        skill_embs_output = temp['node'][s, :]
        
        # print(s.device)
        return skill_embs_output
        # return skill_embs_output


class DySkillHGNN(nn.Module):
    '''Dynamic Skill Heterogeneous Graph Neural Network
    '''

    def __init__(self, hgs, skill_num, skill_embed_dim):
        super().__init__()
        self.n = len(hgs)
        self.hg = hgs[0]
        self.stack_hg = hgs
        self.skill_num = skill_num
        self.hgnn = HeteroHGNN(skill_embed_dim)
        self.edge_attr = [('node', 'is_parent', 'node'), ('node', 'is_child', 'node'), ('node', 'relate', 'node')]
        # .to(list(self.hg.edge_index_dict.values())[0].device)
        self.init_skill_emb = nn.Embedding(skill_num, skill_embed_dim)
        self.skill_embs_temp = []
        self.skill_emb = {"node": self.init_skill_emb.weight}
        # for i in hgs:
        # # self.skill_emb = self.hgnn({"node": self.init_skill_emb.weight}, self.hg.edge_index_dict)
            # self.skill_embs_temp.append(self.skill_emb["node"].clone())
        # self.skill_embs = [skill_emb]
        # self.com_embs = torch.stack(self.com_embs).permute(1, 2, 0)  # [n_com, dim, n_time]
        # self.pos_embs = torch.stack(self.pos_embs).permute(1, 2, 0)  # [n_pos, dim, n_time]
        # self.attn = nn.Parameter(torch.ones(self.n))
        # print(self.skill_embs)
        self.skill_embs = None
        # self.skill_embs = torch.unsqueeze(self.skill_emb["node"],dim=0).repeat(self.n,1,1)
        # self.skill_embs = torch.stack(self.skill_embs_temp) # [n_step, skill_num, embedding_dim]

        self.conv = nn.ModuleDict({''.join(i): normal_conv(skill_embed_dim*2) for i in self.edge_attr})
        self.conv2 = nn.ModuleDict({''.join(i): normal_conv(skill_embed_dim*2) for i in self.edge_attr})
        self.loss_f = nn.MSELoss()
        
        self.L = torch.nn.L1Loss()
        # self.skill_embs = self.init_skill_emb.weight.unsqueeze(0).expand(self.n,-1,-1)

    def forward(self, s, t_s, t_e):
        # if self.skill_embs == None:
        #     self.skill_embs = []
        #     for i in range(self.n):
        #         self.skill_embs.append(self.skill_emb["node"].clone())
        skill_embs_temp = []
        temp = {"node": self.init_skill_emb.weight}
        for i in range(self.n):
            temp = self.hgnn({"node": self.init_skill_emb.weight}, self.stack_hg[i].edge_index_dict)
            skill_embs_temp.append(temp["node"])
        skill_embs = torch.stack(skill_embs_temp) # [n_step, skill_num, embedding_dim]

        # if self.skill_embs.device is not s.device:
        #     self.skill_embs = self.skill_embs.to(s.device)
        # for i in self.skill_embs_temp:
        #     if i.device is not s.device:
        #         i = i.to(s.device)
        # print( torch.eq(self.skill_embs[0,:,:], self.skill_embs[10,:,:]))
        # print(self.stack_hg[t_e.cpu().numpy()[0]])
        # print("start GNN model")
        
        # start_time = torch.min(t_s).cpu().numpy()
        # end_time = torch.max(t_e).cpu().numpy()
        # # print(self.skill_embs[start_time,:,:].shape)
        # temp = {"node": self.skill_embs[start_time,:,:]}
        # combined = []
        # for i in range(self.n):
        #     subgraph_edge_dict = {}
        #     for val in self.edge_attr:
        #         sub_node, sub_edge, mapping, edge_mask= geo_utils.k_hop_subgraph(s, num_hops=1
        #                                         , edge_index=self.stack_hg[i][val]['edge_index'])
        #         subgraph_edge_dict[val] = sub_edge
        #         # print(sub_edge)
        #     # print(self.skill_embs[i, :, :].shape)
        #     # print(subgraph_edge_dict[val].shape)
        #     temp = self.hgnn({"node": self.init_skill_emb.weight}, subgraph_edge_dict)
        #     # print(self.stack_hg[i])
        #     # temp = self.hgnn(temp, self.stack_hg[i].edge_index_dict)
        #     # print(temp)
        #     combined.append(temp["node"])
            
        # temporal_graph_emb = torch.stack(combined)
        # print(temporal_graph_emb.device)
        # print(self.skill_emb["node"].shape)
        # self.skill_embs[t_s]
        # subgraph sampling to generate new graphs
        
        # print("start_sampling")
        # print(temporal_graph_emb[t_e, s, :].shape)
        # print(s.device)
        # print(s.view(1,-1,1).expand(skill_embs.size(0),-1, skill_embs.size(2)).shape)
        # print(skill_embs.gather(1,s.view(1,-1,1).expand(skill_embs.size(0),1, skill_embs.size(2))).shape)
        return skill_embs[:,s,:]
        # return temporal_graph_emb[torch.add(t_e,-1*torch.min(t_s)), s, :]
    def subgraph_sampling(self, s, s_embed, t_s, t_e, current_batch):
        # 1. get subgraph based on s
        # 2. find possbile edge index or remove some possible index in every time interval
        # 3. update to self.stack_hg[i]
        # print(self.stack_hg[0][('node', 'is_parent', 'node')]['edge_index'])
        # print(s_embed.shape)
        end_time = torch.max(t_e)
        # print(s)
        n_node = s.shape[0]
        subgraph_edge_dict = {}
        subgraph_mask = {}
        for val in self.edge_attr[:3]:
        #     print(self.stack_hg[i][val].shape)
            subgraph, _, edge_mask = geo_utils.subgraph(s, self.stack_hg[end_time][val]['edge_index']
                                                        ,relabel_nodes=True, return_edge_mask = True)
        #     # subgraph = geo_utils.subgraph(s, self.stack_hg[i][val])
            subgraph_edge_dict[val] = subgraph
            subgraph_mask[val] = edge_mask
            # print(subgraph)
        # temp = self.hgnn({"node": self.skill_embs[s,:,:]}, subgraph_edge_dict)
        temp = s_embed
        # print(temp.shape)  # [batch_size, embeddings]
        total_loss = 0
        loss = []
        
        for i, val in enumerate(self.edge_attr[:3]):
            # print("here")
            # print(subgraph_edge_dict[val])
            sub_adj = geo_utils.to_dense_adj(subgraph_edge_dict[val], max_num_nodes=n_node)
            # print(temp)
            # print(sub_adj.shape)
            updated_sub_adj = self.find_possible_edges(''.join(val), s, temp, sub_adj)
            update_subgraph_list, edge_attr = geo_utils.dense_to_sparse(updated_sub_adj)
            id = s.clone().int()
            loss.append(self.loss_f(updated_sub_adj, sub_adj).reshape(1)) 
            # print(torch.topk(edge_attr, int(edge_attr.shape[0]*0.1)))
            if current_batch > 2000: 
                # print((edge_attr > 0.5).nonzero().shape)
                # torch.topk(edge_attr,n_node)[1]
                print((edge_attr > 0.5).nonzero().shape)
                # torch.topk(edge_attr, int(0.5*n_node)[1]
                # print(int(0.5*n_node))
                # print(torch.topk(edge_attr, int(0.5*n_node))[1])
                select_num = max(subgraph_edge_dict[val].shape[1], (edge_attr > 0.5).nonzero().shape[0])
                update_subgraph_list = update_subgraph_list[:,  torch.topk(edge_attr, select_num)[1]]
                update_subgraph_list = id[update_subgraph_list]
                # print(update_subgraph_list.shape)
                # update self.hg
                # print(self.stack_hg[i][val]['edge_index'][:, ~subgraph_mask[val]].shape)
                # print(update_subgraph_list.shape)
                self.stack_hg[end_time][val]['edge_index'] = torch.cat((self.stack_hg[end_time][val]['edge_index'][:, ~subgraph_mask[val]], update_subgraph_list), dim=1)
                # print(self.stack_hg[i][val]['edge_index'].shape)
        total_loss = torch.cat(loss)
        total_loss = torch.sum(total_loss)
        return total_loss
    def find_possible_edges(self, edge_type, s, embeddings, sub_adj):
        n, dim = embeddings.shape
        dy_sent = torch.unsqueeze(torch.relu(embeddings), dim=1).repeat(1, n, 1)
        dy_revi = dy_sent.transpose(0,1)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        # print(self.conv[edge_type])
        support = self.conv[edge_type](y, (n, n))
        mask = torch.sigmoid(self.conv2[edge_type](y,(n, n)))
        support = support * mask
        sub_adj = (1-mask) * sub_adj + mask * support
        # sub_adj = F.relu(sub_adj)
        return sub_adj
# ----------------------Utilities----------------------

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class PositionalEncoding(nn.Module):
    '''Positional Encoding for Transformer
    '''

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _get_masked_seq_last(seq: torch.Tensor, l: torch.Tensor, batch_first: bool = True):
    if batch_first:
        assert seq.size(0) == l.size(0)
    else:
        assert seq.size(1) == l.size(0)
    last = []
    for i, x in enumerate(l - 1):
        y = seq[i][x] if batch_first else seq[x][i]
        last.append(y)
    last_tensor = torch.stack(last)
    return last_tensor


def _repeat_seq_dim(seq: torch.Tensor, seq_len: int):
    return seq.unsqueeze(1).repeat(1, seq_len, 1)


class HeteroHGNN(nn.Module):
    '''SKill Heterogeneous Graph Neural Network
    '''

    def __init__(self, skill_embed_dim: int):
        super().__init__()
        self.heterognn = geo_nn.conv.HeteroConv({
            ('node', 'is_parent', 'node'): geo_nn.conv.GATConv(skill_embed_dim, skill_embed_dim,head=3,concat=False, dropout=0.2),
            ('node', 'is_child', 'node'): geo_nn.conv.GATConv(skill_embed_dim, skill_embed_dim,head=3,concat=False, dropout=0.2),
            ('node', 'relate', 'node'): geo_nn.conv.SAGEConv(skill_embed_dim, skill_embed_dim, dropout=0.2),
        }, aggr='mean')
        # self.heterognn_2 = geo_nn.conv.HeteroConv({
        #     ('node', 'is_parent', 'node'): geo_nn.conv.GATConv(skill_embed_dim, skill_embed_dim,dropout=0.2),
        #     ('node', 'is_child', 'node'): geo_nn.conv.GATConv(skill_embed_dim, skill_embed_dim,dropout=0.2),
        #     ('node', 'relate', 'node'): geo_nn.conv.GATConv(skill_embed_dim, skill_embed_dim,dropout=0.2),
        # }, aggr='mean')

    def forward(self, skill_embed, adj):
        temp = self.heterognn(skill_embed, adj)
        # print(temp)
        # temp = self.heterognn_2(temp, adj)
        return temp