import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torchviz import make_dot

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

        self.train_metrics = MetricTracker(*['loss','inflow_loss','outflow_loss'], *["inflow_" + m.__name__ for m in self.metric_ftns], *["outflow_" + m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*['loss','inflow_loss','outflow_loss'], *["inflow_" + m.__name__ for m in self.metric_ftns], *["outflow_" + m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if hasattr(self.model, 'dyskillhgnn'):
            # print(self.model.dyskillhgnn.hg.cuda())
            for hg in self.model.dyskillhgnn.stack_hg:
                hg.to(self.device)
        for (batch_idx, batch) in enumerate(self.data_loader):
            (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e = batch
            # print(d_x_padded[10:20,:])
            # break
            d_x_padded = d_x_padded.to(self.device)
            s_x_padded = s_x_padded.to(self.device)
            d_y = d_y.to(self.device)
            s_y = s_y.to(self.device)
            l = l.to(self.device)
            s = s.to(self.device)
            t_s = t_s.to(self.device)
            t_e = t_e.to(self.device)

            self.optimizer.zero_grad()
            (d_output, s_output) , adj_loss  = self.model((d_x_padded,s_x_padded),l, s, t_s, t_e) # d_output [batch_size, 10], d_y [batch_size]
            # print(adj_loss)
            demand_loss = self.criterion(d_output, target=d_y)
            supply_loss = self.criterion(s_output, target=s_y)
            loss = demand_loss + supply_loss 
            regularized_loss = loss + 1e-3 * adj_loss
            regularized_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('inflow_loss', demand_loss.item())
            self.train_metrics.update('outflow_loss', supply_loss.item())
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_ftns:
                self.train_metrics.update("inflow_" + met.__name__,
                                              met(d_output, d_y, class_num=self.config["hparam"]["class_num"]))
                self.train_metrics.update("outflow_" + met.__name__,
                                          met(s_output, s_y, class_num=self.config["hparam"]["class_num"]))
            
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
            for (batch_idx, batch) in enumerate(self.valid_data_loader):
                (d_x_padded, s_x_padded), (d_y, s_y), l, s, t_s, t_e = batch  # [3,7]
                d_x_padded = d_x_padded.to(self.device)
                s_x_padded = s_x_padded.to(self.device)
                d_y = d_y.to(self.device)
                s_y = s_y.to(self.device)
                l = l.to(self.device)
                s = s.to(self.device)
                t_s = t_s.to(self.device)
                t_e = t_e.to(self.device)

                self.optimizer.zero_grad()
                (d_output, s_output), _  = self.model((d_x_padded.data,s_x_padded.data),l.data, s.data, t_s.data, t_e.data)
                demand_loss = self.criterion(output=d_output, target=d_y)
                supply_loss = self.criterion(output=s_output, target=s_y)
                loss = demand_loss+supply_loss
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('inflow_loss', demand_loss.item())
                self.valid_metrics.update('outflow_loss', supply_loss.item())
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                
                    self.valid_metrics.update("inflow_" + met.__name__,
                                              met(d_output, d_y, class_num=self.config["hparam"]["class_num"]))
                    self.valid_metrics.update("outflow_" + met.__name__,
                                          met(s_output, s_y, class_num=self.config["hparam"]["class_num"]))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

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
