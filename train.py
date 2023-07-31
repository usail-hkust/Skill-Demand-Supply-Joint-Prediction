import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import os


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    
    logger = config.get_logger('train')
   
    # setup data_loader instances
    print('Loading data...')
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    print('Constructing model...')
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], config['device'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = torch.nn.NLLLoss(reduction='none')
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # print(metrics)
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    print("Start Training...")
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    # os.environ["WANDB_MODE"]  = "offline"
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device_id', default=None, type=int,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('-dev', '--device', default=None, type=int,
    #                   help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type defaults target')
    options = [
        CustomArgs(['-n', '--name'], type=str,defaults="defaults", target='name'),
        CustomArgs(['-dev', '--device'], type=str, defaults="3", target='device'),
        CustomArgs(['-wan', '--wandb'], type=bool, defaults=False, target='wandb'),
        # Data related Definition
        CustomArgs(['-bs', '--batch_size'], type=int,defaults=1, target='data_loader;args;batch_size'),
        CustomArgs(['--data_dir'], type=str,defaults="/data/wchao/data/skill_inflow_outflow/", target='data_loader;args;data_dir'),
        CustomArgs(['--dataset'], type=str, defaults="it", target='data_loader;args;dataset'),
        CustomArgs(['-subg', '--subgraph'], type=int, defaults=16956, target='data_loader;args;subgraph_num'),
        # Model
        CustomArgs(['-um', '--univarmodel'], type=str,defaults="Skill_Evolve_Hetero", target='arch;type'),
        CustomArgs(['-m', '--model'], type=str, defaults="static", target='arch;args;model'),
        CustomArgs(['--skill_num'], type=int, defaults=16956, target='arch;args;skill_num'),
        CustomArgs(['--emb_dim'], type=int, defaults=32, target='arch;args;embed_dim'),
        CustomArgs(['--sk_emb_dim'], type=int, defaults=32, target='arch;args;skill_embed_dim'),
        CustomArgs(['--class_num'], type=int, defaults=5, target='arch;args;class_num'),
        CustomArgs(['--layer_num'], type=int, defaults=5, target='arch;args;rnn_layer_num'),
        CustomArgs(['-gcn', '--gcn_layer'], type=int,defaults=2, target='arch;args;gcn_layers'),
        CustomArgs(['--delta'], type=float, defaults=0.1, target='arch;args;delta'),
        CustomArgs(['--sample_node_num'], type=int, defaults=50, target='arch;args;sample_node_num'),
        CustomArgs(['--dropout'], type=float, defaults=0.1, target='arch;args;dropout'),
        CustomArgs(['-hyp', '--hyperdecode'], type=bool,defaults=False, target='arch;args;hyperdecode'),
        CustomArgs(['-hypcomb', '--hypcomb'], type=str,defaults="add", target='arch;args;hypcomb'),
        
        
        CustomArgs(['-lr', '--learning_rate'], type=float,defaults=1e-4, target='optimizer;args;lr')
        
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
