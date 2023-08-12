import os
import numpy as np


class Config_MAE_fMRI: # back compatibility
    pass

class Config_MBM_fMRI(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Parameters for GAE
        self.num_iter = 50
        self.wd = 5e-4
        self.lr = 0.001
        self.batch_size = 128
        self.feature_num  = 900
        self.gcn_hidden_layer = [1024, 512]
        self.conv_kernels = [16, 32]
        self.gnn_type = 'GraphConv'
        self.masking = False
        self.masking_ratio = 0.2
        self.time = ''