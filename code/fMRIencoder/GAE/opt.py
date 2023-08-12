import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config
from config import Config_MBM_fMRI

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Implementation of GAE')
        parser.add_argument('--num_iter', type=int, default=50, help='number of epochs for training')
        parser.add_argument('--lr',  type=float,default=0.0005, help='initial learning rate')
        parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        
        parser.add_argument('--feature_num', type=int, default=900, help='hidden units of gconv layer 1')
        parser.add_argument('--gcn_hidden_layer', type=int, default=[1024, 512], nargs='+',help='hidden units of gconv layers')
        parser.add_argument('--conv_kernels', type=int, default=[16, 32], nargs='+',help='hidden units of conv layers')
        parser.add_argument('--gnn_type', type=str, default='GraphConv', help='type of gnn')
        parser.add_argument('--masking', type=bool, default=False, help='utilization of feature masking')
        parser.add_argument('--masking_ratio', type=float, default=0.2, help='ratio of feature masking')

        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.args = args
        self.config = Config_MBM_fMRI()
        self.config = self.update_config(args, self.config)

    def print_args(self):
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}: {}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")

    def update_config(self, args, config):
        for attr in config.__dict__:
            if hasattr(args, attr):
                if getattr(args, attr) != None:
                    setattr(config, attr, getattr(args, attr))

        return config

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.config

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False