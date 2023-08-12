import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
from utils import *
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


class NSD_Dataset(Dataset):
    def __init__(self, data_dir, subject_num, type, config):
        """
        ---------------------------------------------------
        Parameter
        data_dir : dataset base directory
        subject_num : subject number 01, 02, 03, ...
        type : "train", "train_ae", "test", "val", "val_ae"
        ---------------------------------------------------
        Dataset structure
         {
            "Image": 3D Tensor of Image (256,256,3), uint8
            "fMRI": { x1 : 1d Tensor of signal (756), float64
                      x2 : 1d Tensor of signal (594), float64
                      x3 : 1d Tensor of signal (599), float64
                      x4 : 1d Tensor of signal (834), float64
                      x5 : 1d Tensor of signal (541), float64
                      x6 : 1d Tensor of signal (646), float64
                      x7 : 1d Tensor of signal (687), float64
            },
            "Image_Index": int : image index number
        }
        ---------------------------------------------------
        """

        image_extension = '.npy'
        self.subj = subject_num  
        self.type = type
        self.data_dir = data_dir + '/' + self.type + '/subj{}'.format(self.subj)
        self.image_dir = data_dir + '/image_files/'

        npy_files = [
            'subj{}_V1d_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V1v_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V2d_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V2v_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V3d_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V3v_sorted_{}.npy'.format(self.subj, self.type),
            'subj{}_V4_sorted_{}.npy'.format(self.subj, self.type)
        ]

        csv_file = 'subj{}_concatenated_csv_{}.csv'.format(self.subj, self.type)

        npy_datas = []
        for npy_file in npy_files:
            npy_path = os.path.join(self.data_dir, npy_file)
            npy_data = np.load(npy_path)
            npy_datas.append(npy_data)

        csv_path = os.path.join(self.data_dir, csv_file)
        csv_data = pd.read_csv(csv_path)
        
        self.prepared_data = []
        for i in range(len(csv_data)):
            fMRI_data = {}
            for j, npy_file in enumerate(npy_files):
                fMRI_key = npy_file.split('_')[1]
                size = config.feature_num - len(npy_datas[j][i])

                padded_array = np.pad(npy_datas[j][i], (size// 2, size//2 + size % 2), 'wrap')
                fMRI_data[fMRI_key] =torch.from_numpy(padded_array).float().view(1, -1)
            
            adj = compute_cosine_adj(fMRI_data)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)

            prepared_item = {
                "fMRI": Data(x1 = fMRI_data["V1d"], x2 = fMRI_data["V1v"], x3 = fMRI_data["V2d"], x4 = fMRI_data["V2v"], 
                    x5= fMRI_data["V3d"], x6 =fMRI_data["V3v"], x7=fMRI_data["V4"], edge_index=edge_index, edge_attr=edge_attr)
            }

            self.prepared_data.append(prepared_item)
            
    def __len__(self):
        return len(self.prepared_data)

    def __getitem__(self, idx):
        return self.prepared_data[idx]