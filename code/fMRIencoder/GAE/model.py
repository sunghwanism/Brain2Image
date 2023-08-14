import torch
import torch.nn as nn
import torch.nn.functional as func
from torch_geometric.nn import GCNConv, GraphConv, GATConv
import copy

class GAE(torch.nn.Module):
    def __init__(self, feature_num, gcn_hidden_layer, conv_kernels, gnn_type = "GraphConv", masking = False, masking_ratio = .2):
        super(GAE, self).__init__()
        self.feature_num = feature_num
        self.gcn_hidden_layer = copy.deepcopy(gcn_hidden_layer) # insert 에서 값이 변하는 이슈로 deepcopy 처리
        self.gcn_hidden_layer.insert(0, feature_num)
        self.conv_kernels = copy.deepcopy(conv_kernels)
        self.conv_kernels.insert(0, 1)
        self.gcn_num_layer = len(self.gcn_hidden_layer) - 1
        self.conv_num_layer = len(self.conv_kernels) - 1
        self.masking = masking
        self.masking_ratio = masking_ratio

        self.linear_sizes = [756, 594, 599, 834, 541, 646, 687]
        
        #Encoder MLP
        self.elin = nn.ModuleList([nn.Linear(size, self.feature_num) for size in self.linear_sizes])

        #Decoder MLP
        self.dlin = nn.ModuleList([nn.Linear(self.feature_num, size) for size in self.linear_sizes])

        #GAE
        self.encoder = nn.ModuleList() # gnn
        self.ebn = nn.ModuleList() # batchnorm
        self.decoder = nn.ModuleList() # gnn
        self.dbn = nn.ModuleList() # batchnorm
        self.enconv = nn.ModuleList() # cnn
        self.deconv = nn.ModuleList() # cnn

        for i in range(self.gcn_num_layer):
            if gnn_type == "GraphConv":
                self.encoder.append(GraphConv(self.gcn_hidden_layer[i], self.gcn_hidden_layer[i+1]))
                self.decoder.append(GraphConv(self.gcn_hidden_layer[self.gcn_num_layer-i], self.gcn_hidden_layer[self.gcn_num_layer-(i+1)]))
            elif gnn_type == "GCNConv":
                self.encoder.append(GCNConv(self.gcn_hidden_layer[i], self.gcn_hidden_layer[i+1]))
                self.decoder.append(GCNConv(self.gcn_hidden_layer[self.gcn_num_layer-i], self.gcn_hidden_layer[self.gcn_num_layer-(i+1)]))
            elif gnn_type == "GATConv":
                self.encoder.append(GATConv(self.gcn_hidden_layer[i], self.gcn_hidden_layer[i+1]))
                self.decoder.append(GATConv(self.gcn_hidden_layer[self.gcn_num_layer-i], self.gcn_hidden_layer[self.gcn_num_layer-(i+1)]))

            self.ebn.append(nn.BatchNorm1d(self.gcn_hidden_layer[i+1]))
            self.dbn.append(nn.BatchNorm1d(self.gcn_hidden_layer[self.gcn_num_layer-(i+1)]))

        for i in range(self.gcn_num_layer): 
            self.enconv.append(nn.Conv1d(self.conv_kernels[i], self.conv_kernels[i+1], 3, padding=1))
            self.deconv.append(nn.ConvTranspose1d(self.conv_kernels[self.gcn_num_layer-i], self.conv_kernels[self.gcn_num_layer-(i+1)], 3, padding=1))

    ##### encoder MLP
    def encodeMLP(self, data):
        temp_x = [data[f'x{i+1}'] for i in range(7)]
        raw_x = copy.deepcopy(temp_x)

        # 각 x를 크기에 맞는 linear에 넣어서 크기를 맞춰주고, 하나의 merged_x 변수로 만들어줌
        if self.masking == True:     
            merged_x = [func.relu(self.elin[i](func.dropout(x, p = self.masking_ratio, training=self.training))) for i, x in enumerate(raw_x)]
        else:
            merged_x = [func.relu(self.elin[i](x)) for i, x in enumerate(raw_x)]

        # merged x 는 모든 x1, 모든 x2 ... 순으로 이루어져있음
        # 그래프 처리를 위해 x1[0], x2[0], ... 순으로 텐서를 재구성해줌
        for i in range(0, data.x1.shape[0]):
            if i == 0:
                x = torch.stack([merged_x[j][i] for j in range(7)], dim=0)
            else:
                y = torch.stack([merged_x[j][i] for j in range(7)], dim=0)
                x = torch.cat((x, y), dim = 0)

        return x
        
    #### feature padding
    def feature_arrangement(self, data):
        temp_x = [data[f'x{i+1}'] for i in range(7)]
        raw_x = copy.deepcopy(temp_x)

        if self.masking == True:     
            merged_x = [func.dropout(x, p = self.masking_ratio, training=self.training) for i, x in enumerate(raw_x)]
        else:
            merged_x = [x for i, x in enumerate(raw_x)]
        
        # merged x 는 모든 x1, 모든 x2 ... 순으로 이루어져있음
        # 그래프 처리를 위해 x1[0], x2[0], ... 순으로 텐서를 재구성해줌
        for i in range(0, data.x1.shape[0]):
            if i == 0:
                x = torch.stack([merged_x[j][i] for j in range(7)], dim=0)
            else:
                y = torch.stack([merged_x[j][i] for j in range(7)], dim=0)
                x = torch.cat((x, y), dim = 0)

        return x

    ##### encoder GNN
    def encodeGNN(self, x, edge_index, edge_attr):
        for i in range(self.gcn_num_layer):
            x = self.encoder[i](x, edge_index, edge_attr)
            if i < self.gcn_num_layer - 1:
                x = func.relu(self.ebn[i](x))

        return x

    ##### encoder CNN
    def encodeCNN(self, x):
        x = x.squeeze().unsqueeze(1)
        for i in range(self.conv_num_layer):
            x = self.enconv[i](x)
            if i < self.conv_num_layer - 1:
                x = func.relu(x)

        return x

    ##### encoder CNN
    def decodeCNN(self, x):
        for i in range(self.conv_num_layer):
            x = self.deconv[i](x)
            if i < self.conv_num_layer - 1:
                x = func.relu(x)
            
        x = x.squeeze()
        
        return x
    
    ##### decoder GNN
    def decodeGNN(self, x, edge_index, edge_attr):
        for i in range(self.gcn_num_layer):
            x = self.decoder[i](x, edge_index, edge_attr)
            if i < self.gcn_num_layer - 1:
                x = func.relu(self.dbn[i](x))
        
        return x
    
    def matching_size(self, x):
        groups = []
        for i in range(7):
            group = []
            for j in range(i, x.shape[0], 7):
                group.append(j)
            groups.append(group)

        grouped_x = [x[group] for group in groups]

        x = torch.cat([grouped_x[i] for i in range(7)], dim=1)
        
        return x
    
    ##### GAE
    def forward(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        x = self.feature_arrangement(data)
        z = self.encodeGNN(x, edge_index, edge_attr)
        zconv = self.encodeCNN(z)
        
        # zconv.shape => ((batch * 7) * 32 * 512)

        zdeconv = self.decodeCNN(zconv)
        h = self.decodeGNN(zdeconv, edge_index, edge_attr)
        x = self.matching_size(x)
        h = self.matching_size(h)
    
        return x, h
    
    def load_checkpoint(self, state_dict):
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
