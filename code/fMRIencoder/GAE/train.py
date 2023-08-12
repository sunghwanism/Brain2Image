import torch
import torch.nn.functional as func
from torch_geometric.loader import DataLoader
from model import GAE
from dataset import NSD_Dataset
from utils import *
from opt import * 

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def GAE_train(loader):
    model.train()
    train_loss_all = 0
    train_mae_all = 0
    pred = []
    real = []
    
    for data in loader:
        data = data["fMRI"].to(device)
        optimizer.zero_grad()
        realX, predX = model(data)

        # Loss 계산
        train_loss = func.mse_loss(predX, realX)
        train_mae = func.l1_loss(predX, realX)
        train_loss.backward()
        train_loss_all += data.num_graphs * train_loss.item()
        train_mae_all += data.num_graphs * train_mae.item()
        optimizer.step()
        
        pred.append(predX)
        real.append(realX)

    predXs = torch.cat(pred, dim=0).cpu().detach().numpy()
    realXs = torch.cat(real, dim=0).cpu().detach().numpy()
    corr = correlation(predXs, realXs)

    return train_loss_all / len(train_dataset), train_mae_all / len(train_dataset), corr, realXs[0:3], predXs[0:3]

def GAE_test(loader):
    model.eval()
    val_loss_all = 0
    val_mae_all = 0
    pred = []
    real = []

    for data in loader:
        data = data["fMRI"].to(device)
        realX, predX = model(data)

        # Loss 계산
        val_loss = func.mse_loss(predX, realX)
        val_mae = func.l1_loss(predX, realX)
        val_loss_all += data.num_graphs * val_loss.item()
        val_mae_all += data.num_graphs * val_mae.item()

        pred.append(predX)
        real.append(realX)

    predXs = torch.cat(pred, dim=0).cpu().detach().numpy()
    realXs = torch.cat(real, dim=0).cpu().detach().numpy()
    corr = correlation(predXs, realXs)
    
    return val_loss_all / len(val_dataset), val_mae_all / len(val_dataset), corr, realXs[0:3], predXs[0:3]

config = OptInit().initialize()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder_paths = ["./results", "./results/preds", "./results/figures", "./results/models"]
for folder in folder_paths:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
print('  Loading dataset ...')
data_dir = 'C:/Users/hyewon/Documents/GitHub/GAE/data/data_renew_no_z_score/data'
subject_num = "01"
train_dataset = NSD_Dataset(data_dir, subject_num, "train_ae", config)
val_dataset = NSD_Dataset(data_dir, subject_num, "val_ae", config)

model = GAE(feature_num = config.feature_num, gcn_hidden_layer=config.gcn_hidden_layer, 
            conv_kernels= config.conv_kernels, gnn_type=config.gnn_type, masking = config.masking, masking_ratio = config.masking_ratio).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Train & Validation
for epoch in range(config.num_iter):
    t_loss, t_mae, t_corr, t_real, t_pred = GAE_train(train_loader)
    v_loss, v_mae, v_corr, v_real, v_pred = GAE_test(val_loader)
    print('[{:03d}]  Train MSE: {:.4f},  Val MSE: {:.4f}'.format(epoch + 1, t_loss, v_loss))

save_model(config, epoch, model, optimizer,'./results/models/')
save_voxel_csv(config, t_real, t_pred, v_real, v_pred)

print('\nFinish Training')
print('[Train]  MSE: {:.4f},  MAE: {:.4f},  Correlation: {:.4f}'.format(t_loss, t_mae, t_corr))
print('[Val]  MSE: {:.4f},  MAE: {:.4f},  Correlation: {:.4f}'.format(v_loss, v_mae, v_corr))

save_results(config, t_loss, t_mae, t_corr, v_loss, v_mae, v_corr)
for i in range(0, 3): vizPlot(config, i)