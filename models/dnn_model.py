import torch
import torch.nn as nn
from ..layers import DNN

class DnnModel(nn.Module):
    def __init__(self):
        super(DnnModel, self).__init__()

        config = default_config()
        config['input_dim'] = config['feature_num'] * config['feature_dim']

        self.decr_dim = []
        for (name, dim) in features:
            self.decr_dim.append(torch.nn.Linear(dim, config['feature_dim']))

        self.dnn = DNN(config)
        self.relu = torch.nn.ReLU()
        self.lin = torch.nn.Linear(config['output_dim'], 1)
        self.sigm = torch.nn.Sigmoid()

        self.loss_f = torch.nn.BCELoss()
    
    def forward(self, features_x):
        batch_size = features_x[0].size(0)
        for i, f in enumerate(features_x):
            features_x[i] = self.decr_dim[i](f.float())
        
        input_features = torch.cat(features_x, -1)
        
        dnn_output = self.dnn(input_features)

        relu_output = self.relu(dnn_output)

        lin_out = self.lin(relu_output)

        output = self.sigm(lin_out)
        return output
    
    def loss(self, y_pred, y_true):
        return self.loss_f(y_pred, y_true)
