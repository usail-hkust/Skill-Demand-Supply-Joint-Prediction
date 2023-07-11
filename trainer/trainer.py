import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torchviz import make_dot
import torch_geometric.utils as geo_utils
import wandb

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.aux_loss = torch.nn.CrossEntropyLoss()
        self.train_metrics = MetricTracker(*['loss','demand_loss','supply_loss',"auxiliary_loss"],*["joint_" + self.metric_ftns[0].__name__], *["demand_" + m.__name__ for m in self.metric_ftns], *["supply_" + m.__name__ for m in self.metric_ftns],*["total_" + m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*['loss','demand_loss','supply_loss',"auxiliary_loss"],*["joint_" + self.metric_ftns[0].__name__], *["demand_" + m.__name__ for m in self.metric_ftns], *["supply_" + m.__name__ for m in self.metric_ftns],*["total_" + m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        columns = ["id","pred_demand","pred_supply","label_demand","label_supply"]
        for i in range(5):
            columns.append("score_"+str(i))
        if config["wandb"] == True:
            self.test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")
            self.test_table = wandb.Table(columns=columns)
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if hasattr(self.model, 'dyskillhgnn'):
            self.model.dyskillhgnn.hg.to(self.device)
            # print(self.model.dyskillhgnn.hg.cuda())
            # for hg in self.model.dyskillhgnn.stack_hg:
            #     hg.to(self.device)
                # for i in hg:
                #     i.to(self.device)

        for (batch_idx, batch) in enumerate(self.data_loader):
            (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e, gap = batch
            skill_semantic_embed = self.data_loader.skill_semantic_emb.to(self.device)
            demand_graph_input = self.data_loader.graphdata.to(self.device)
            auxiliary_label = self.data_loader.dataset.classification.to(self.device)
            # demand_graph_target = self.data_loader.graphdata[t_e+1].to(self.device)
            # demand_graph_target = geo_utils.to_dense_adj(edge_index=demand_graph_target.edge_index, edge_attr=demand_graph_target.edge_attr).squeeze()
            # supply_graph = self.data_loader.supply_graph
            # graph = graph.to(self.device)
            comm = None
            # comm = []
            # print(len(comm))
            # print(graph)
            # print(d_x_padded.shape)
            # break
            d_x_padded = d_x_padded.to(self.device).squeeze()
            s_x_padded = s_x_padded.to(self.device).squeeze()
            d_y = d_y.to(self.device).squeeze()
            s_y = s_y.to(self.device).squeeze()
            l = l.to(self.device)
            s = s.to(self.device)
            t_s = t_s.to(self.device)
            t_e = t_e.to(self.device)
            gap = gap.to(self.device)
            # print(d_x_padded.shape)
            self.optimizer.zero_grad()
            (d_output, s_output), gen_graph, model_loss, pred = self.model((d_x_padded, s_x_padded), l, s, t_s, t_e, demand_graph_input, comm, skill_semantic_embed, gap) # d_output [batch_size, 10], d_y [batch_size]
            # print(gen_graph.shape)
            # print(demand_graph_target.shape)
            demand_loss = self.criterion(d_output, target=d_y)
            supply_loss = self.criterion(s_output, target=s_y)
            # auxiliary_loss = self.aux_loss(pred, auxiliary_label)
            joint_loss = torch.sqrt((demand_loss-supply_loss)**2+1e-8).mean()
            joint_loss = 0
            supply_loss = supply_loss.mean()
            demand_loss = demand_loss.mean()
            # print(demand_loss)
            auxiliary_loss = torch.zeros(1)
            # adj_loss_1 = torch.nn.functional.mse_loss(gen_graph[:d_x_padded.shape[0], :d_x_padded.shape[0]], demand_graph_target[:d_x_padded.shape[0],:d_x_padded.shape[0]])
            # adj_loss_2 = torch.nn.functional.mse_loss(gen_graph[d_x_padded.shape[0]:, d_x_padded.shape[0]:], demand_graph_target[d_x_padded.shape[0]:, d_x_padded.shape[0]:])
            # adj_loss = adj_loss_1 + adj_loss_2
            adj_loss=0
            # model_loss, auxiliary_loss
            # loss = demand_loss + supply_loss + 1e-2 * auxiliary_loss
            # print((demand_loss* supply_loss))
            loss = (demand_loss + supply_loss).mean() + 1e-1*joint_loss
            # + 
            regularized_loss = loss + 1e-3 * adj_loss
            regularized_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            


            self.train_metrics.update('demand_loss', demand_loss.item())
            self.train_metrics.update('supply_loss', supply_loss.item())
            # self.train_metrics.update('adj_loss', adj_loss.item())
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update("auxiliary_loss", auxiliary_loss.item())
            self.train_metrics.update("joint_" + self.metric_ftns[0].__name__,
                                              self.metric_ftns[0](torch.stack([d_output, s_output],dim=1), torch.stack([d_y, s_y],dim=1), class_num=self.config["hparam"]["class_num"], joint=True))
            for met in self.metric_ftns:
                self.train_metrics.update("demand_" + met.__name__,
                                              met(d_output, d_y, class_num=self.config["hparam"]["class_num"]))
                self.train_metrics.update("supply_" + met.__name__,
                                          met(s_output, s_y, class_num=self.config["hparam"]["class_num"]))
                self.train_metrics.update("total_" + met.__name__,
                                          met(torch.cat([d_output, s_output],dim=0), torch.cat([d_y, s_y],dim=0), class_num=self.config["hparam"]["class_num"]))
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            demand_loss_sum = 0
            supply_loss_sum = 0
            loss_sum = 0
            auxiliary_loss_sum = 0
            joint_loss_sum = 0
            s_output_list = []
            s_label_list = []
            d_output_list = []
            d_label_list = []
            for (batch_idx, batch) in enumerate(self.valid_data_loader):
                (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e, gap = batch
                # print(t_e)
                skill_semantic_embed = self.data_loader.skill_semantic_emb.to(self.device)
                demand_graph_input = self.data_loader.graphdata.to(self.device)
                auxiliary_label =self.data_loader.dataset.classification.to(self.device)
                # demand_graph_target = self.data_loader.graphdata[t_e+1].to(self.device)
                # supply_graph = self.data_loader.supply_graph
                # graph = graph.to(self.device)
                comm = None
                # comm = []
                # print(len(comm))
                # print(graph)
                # print(d_x_padded.shape)
                # break
                d_x_padded = d_x_padded.to(self.device).squeeze()
                s_x_padded = s_x_padded.to(self.device).squeeze()
                d_y = d_y.to(self.device).squeeze()
                s_y = s_y.to(self.device).squeeze()
                l = l.to(self.device)
                s = s.to(self.device)
                t_s = t_s.to(self.device)
                t_e = t_e.to(self.device)
                gap = gap.to(self.device)
                # print(d_x_padded.shape)
                self.optimizer.zero_grad()
                (d_output, s_output) , gen_graph, model_loss, pred = self.model((d_x_padded, s_x_padded), l, s, t_s, t_e, demand_graph_input, comm, skill_semantic_embed, gap) # d_output [batch_size, 10], d_y [batch_size]
                # print(d_output.shape)
                # print(d_y.shape)
                demand_loss = self.criterion(d_output, target=d_y)
                supply_loss = self.criterion(s_output, target=s_y)
                joint_loss = torch.sqrt((demand_loss-supply_loss)**2+1e-8).mean()
                # joint_loss = 0
                supply_loss = supply_loss.mean()
                demand_loss = demand_loss.mean()
                # print(demand_loss)
                # auxiliary_loss = self.aux_loss(pred, auxiliary_label)
                auxiliary_loss = torch.zeros(1)
                # combined_data = torch.cat([torch.arange(d_output.shape[0],device=self.device), d_output, s_output,d_y, s_y]).T
                
                # adj_loss = torch.nn.functional.mse_loss(gen_graph, demand_graph_target)
                adj_loss = 0
                loss = (demand_loss + supply_loss).mean() + 1e-2*joint_loss
                # print(loss) 
                # + torch.sqrt((demand_loss-supply_loss)**2)

                demand_loss_sum+=demand_loss
                supply_loss_sum+=supply_loss
                loss_sum+=loss
                joint_loss_sum+=joint_loss

                s_output_list.append(s_output)
                s_label_list.append(s_y)
                d_output_list.append(d_output)
                d_label_list.append(d_y)
                
            self.writer.set_step(epoch, 'valid')
            self.valid_metrics.update('demand_loss', demand_loss_sum.item()/len(self.valid_data_loader))
            self.valid_metrics.update('supply_loss', supply_loss_sum.item()/len(self.valid_data_loader))
            # self.train_metrics.update('adj_loss', adj_loss.item())
            self.valid_metrics.update('loss', loss_sum.item()/len(self.valid_data_loader))
            self.valid_metrics.update('auxiliary_loss', joint_loss.item()/len(self.valid_data_loader))

            s_output = torch.cat(s_output_list,dim=0)
            s_y = torch.cat(s_label_list,dim=0)
            d_output = torch.cat(d_output_list,dim=0)
            d_y = torch.cat(d_label_list,dim=0)
            self.valid_metrics.update("joint_" + self.metric_ftns[0].__name__,
                                              self.metric_ftns[0](torch.stack([d_output, s_output],dim=1), torch.stack([d_y, s_y],dim=1), class_num=self.config["hparam"]["class_num"], joint=True))
            for met in self.metric_ftns:
                self.valid_metrics.update("demand_" + met.__name__,
                                            met(d_output, d_y, class_num=self.config["hparam"]["class_num"]))
                self.valid_metrics.update("supply_" + met.__name__,
                                        met(s_output, s_y, class_num=self.config["hparam"]["class_num"]))
                self.valid_metrics.update("total_" + met.__name__,
                                        met(torch.cat([d_output, s_output],dim=0), torch.cat([d_y, s_y],dim=0), class_num=self.config["hparam"]["class_num"]))
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        if epoch%100 == 0:
            pred_d = torch.argmax(d_output, dim=-1).cpu().tolist()
            pred_s = torch.argmax(s_output, dim=-1).cpu().tolist()
            d_y = d_y.cpu().tolist()
            s_y = s_y.cpu().tolist()
            
            # columns = ["id","pred_demand","pred_supply","label_demand","label_supply"]
            if False: # self.config["wandb"] & 
                for i in range(len(pred_d)):
                    d_o = d_output[i,:].cpu().tolist()
                    self.test_table.add_data(i, pred_d[i],pred_s[i],d_y[i],s_y[i],*d_o)
                # self.test_table.add_column(name="pred_demand", data=pred_d)
                # self.test_table.add_column(name="pred_supply", data=pred_s)
                # self.test_table.add_column(name="label_demand", data=d_y)
                # self.test_table.add_column(name="label_supply", data=s_y)
                # self.test_table.add_column(name="skill_dict", data=[i for i in range(d_output.shape[0])])
                
                    self.test_data_at.add(self.test_table, "predictions")
                    wandb.run.log_artifact(self.test_data_at)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
