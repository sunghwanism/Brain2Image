import numpy as np
import pandas as pd
import csv
import os
import torch
import matplotlib.pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

def compute_cosine_adj(fMRI_data):
     A = np.full((7, 7), 0)
     keys = list(fMRI_data.keys())

     for i in range(len(fMRI_data)):
          for j in range(len(fMRI_data)):
               tensor_a = fMRI_data[keys[i]]
               tensor_b = fMRI_data[keys[j]]
               A[i, j] = torch.cosine_similarity(tensor_a, tensor_b)

     return A

def save_voxel_csv(config, t_real, t_pred, test_real, test_pred):
     t_real, t_pred = pd.DataFrame(t_real), pd.DataFrame(t_pred)
     test_real, test_pred = pd.DataFrame(test_real), pd.DataFrame(test_pred)

     folder_path = f'./results/preds/{config.time}'
     
     if not os.path.exists(folder_path):
          os.makedirs(folder_path)

     t_real.to_csv(f'./{folder_path}/train_real.csv')
     t_pred.to_csv(f'./{folder_path}/train_pred.csv')
     test_real.to_csv(f'./{folder_path}/test_real.csv')
     test_pred.to_csv(f'./{folder_path}/test_pred.csv')

def save_results(config, t_loss, t_mae, t_corr, v_loss, v_mae, v_corr):
     filename = "./results/gae_conv.csv"
     file_exists = os.path.isfile(filename)
     cols = list(vars(config).keys())
     cols += ["train mse loss", "train mae", "train corr", "val mse loss", "val mae", "val corr"]

     vals = list(vars(config).values())
     vals += [t_loss, t_mae, t_corr, v_loss, v_mae, v_corr]

     with open(filename, "a", newline="") as csvfile:
          writer = csv.writer(csvfile)

          if not file_exists:
               writer.writerow(cols)
               writer.writerow(vals)
     
          else:
               writer.writerow(vals)

def vizPlot(config, idx):
    folder_path = f'./results/figures/{config.time}'
    if not os.path.exists(folder_path):
         os.makedirs(folder_path)
    test_pred = pd.read_csv(f"./results/preds/{config.time}/test_pred.csv", index_col=0)
    test_real = pd.read_csv(f"./results/preds/{config.time}/test_real.csv", index_col=0)
    title = ["V1d", "V1v", "V2d", "V2v", "V3d", "V3v", "V4"]
    fig, axs = plt.subplots(7, 1, figsize=(30,30))
    fig.tight_layout()
    x = test_pred.columns
    y1 = test_real.iloc[idx]
    y2 = test_pred.iloc[idx]
    s = 0
    for i in range(7):
        axs[i].plot(x[s:s+config.feature_num], y1[s:s+config.feature_num], label='Real', alpha=1.0, color='#55ACFF', linewidth=1.5)
        axs[i].plot(x[s:s+config.feature_num], y2[s:s+config.feature_num], label='Pred', alpha=0.7, color='#FF6865', linewidth=1.5)
        axs[i].set_title(title[i],  fontsize=20)
        axs[i].set_xticks([])
        s += config.feature_num
        #plt.ylim(-30, 40)
        if i == 0: axs[i].legend(loc='upper right',  fontsize=20)
    fig.savefig(f'./{folder_path}/{idx}.png', dpi=200, bbox_inches='tight', pad_inches=0)

def correlation(predXs, realXs):
     corrs = []

     for row1, row2 in zip(predXs, realXs):
          correlation, _ = stats.pearsonr(row1, row2)
          corrs.append(correlation)

     mean = np.mean(corrs)
     
     return mean

def save_model(config, epoch, model, optimizer, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        #'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
