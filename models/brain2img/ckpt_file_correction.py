import torch
import os

model_path = os.path.join('/home/leehu/project/brain2image/brain2image/results/generation/04-07-2023-21-31-52/checkpoint.pth')
sd = torch.load(model_path, map_location='cpu')
print(sd['config'].data_dir)
# sd['config'].data_dir = '/NFS/Users/jhlee/NSD/'
# print(sd['config'].data_dir)

# torch.save(sd, model_path)