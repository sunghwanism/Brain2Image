import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import tqdm

class NSD_Dataset(Dataset):
    def __init__(self, data_dir, subject_num, type):
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
            "fMRI": { V1d : 1d Tensor of signal (756), float64
                      V1v : 1d Tensor of signal (594), float64
                      V2d : 1d Tensor of signal (599), float64
                      V2v : 1d Tensor of signal (834), float64
                      V3d : 1d Tensor of signal (541), float64
                      V3v : 1d Tensor of signal (646), float64
                      V4 : 1d Tensor of signal (687), float64
            },
            "Image_Index": int : image index number
        }
        ---------------------------------------------------
        """

        image_extension = '.npy'
        self.subj = subject_num  
        self.type = type
        self.data_dir = data_dir + '/data/' + self.type + '/subj{}'.format(self.subj)
        self.image_dir = data_dir + '/image_files/'
        self.num_voxels = 848*7

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

        data = []
        patch_size=16
        max_length=int(self.num_voxels/7)
        for npy_file in tqdm.tqdm(npy_files):
            npy_path = os.path.join(self.data_dir, npy_file)
            npy_data = np.load(npy_path)
            npy_data = np.pad(npy_data, ((0,0),(0, patch_size-npy_data.shape[1]%patch_size)), 'wrap')
            if npy_data.shape[-1] == max_length:
                npy_data
    
            npy_data = np.pad(npy_data, ((0,0), (0, max_length - npy_data.shape[-1])), 'wrap')
            data.append(npy_data)

        csv_path = os.path.join(self.data_dir, csv_file)
        csv_data = pd.read_csv(csv_path)
        
        self.prepared_data = []
        for i in tqdm.tqdm(range(len(csv_data))):
            image_index = int(csv_data.loc[i, 'index'])
            image_path = self.image_dir + str(image_index) + image_extension
            image_array = np.load(image_path)
            image_array = (image_array / 255.0) * 2.0 - 1.0

            fMRI_data = {}
            
            for j, npy_file in enumerate(npy_files):
                fMRI_key = npy_file.split('_')[1]
                fMRI_data[fMRI_key] = data[j][i]
                
            fmri_concat = np.empty(shape=(1,0))
            for key, value in fMRI_data.items():
                fmri_concat = np.concatenate((fmri_concat, np.expand_dims(value, axis=0)), axis=-1)
            
            prepared_item = {
                "Image": image_array,
                "fMRI": fmri_concat,
                "Image_Index": image_index
            }

            self.prepared_data.append(prepared_item)

    def __len__(self):
        return len(self.prepared_data)

    def __getitem__(self, idx):
        return self.prepared_data[idx]