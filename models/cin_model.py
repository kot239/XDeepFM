import torch
import torch.nn as nn
from ..layers import CIN


class CinModel(nn.Module):
    def __init__(self):
        super(CinModel, self).__init__()
        
        config = default_config()

        self.feat_size = config['feature_num']
        self.emb_size = config['feature_dim']

        self.decr_dim = []
        for (name, dim) in features:
            self.decr_dim.append(torch.nn.Linear(dim, self.emb_size))
        
        self.cin = CIN(config)
        self.relu = torch.nn.ReLU()
        self.lin = torch.nn.Linear(sum(config['cin_hidden_layers']), 1)
        self.sigm = torch.nn.Sigmoid()

        self.loss_f = torch.nn.BCELoss()
    
    def forward(self, features_x):
        batch_size = features_x[0].size(0)
        for i, f in enumerate(features_x):
            features_x[i] = self.decr_dim[i](f.float())

        input_features = torch.cat(features_x, 1).reshape(batch_size, self.feat_size, self.emb_size)
        
        cin_output = self.cin(input_features)

        relu_output = self.relu(cin_output)

        lin_out = self.lin(relu_output)

        output = self.sigm(lin_out)
        return output
    
    def loss(self, y_pred, y_true):
        return self.loss_f(y_pred, y_true)
