import torch
import torch.nn as nn
from ..layers import CIN, DNN
from ..utils import default_config

class XDeepFMModel(nn.Module):
    def __init__(self, features):
        super(XDeepFMModel, self).__init__()    

        self.config = default_config(features)
        
        self.config['input_dim'] = self.config['feature_num'] * self.config['feature_dim']

        self.decr_dim = []
        for (name, dim) in features:
            self.decr_dim.append(torch.nn.Linear(dim, self.config['feature_dim']))

        self.cin = CIN(config=self.config)
        self.dnn = DNN(config=self.config)
        self.lin = torch.nn.Linear(sum([list(feat) for feat in zip(*features)][1]), self.config['output_dim'])

        self.w_cin = torch.nn.Linear(sum(self.config['cin_hidden_layers']), 1)
        self.w_dnn = torch.nn.Linear(self.config['output_dim'], 1)
        self.w_lin = torch.nn.Linear(self.config['output_dim'], 1)

        self.sigm = torch.nn.Sigmoid()

        self.loss_f = torch.nn.BCELoss()
    

    def forward(self, features_x):
        batch_size = features_x[0].size(0)

        lin_input = torch.cat(features_x, -1)

        for i, f in enumerate(features_x):
            features_x[i] = self.decr_dim[i](f.float())

        dnn_input = torch.cat(features_x, -1)
        cin_input = dnn_input.reshape(batch_size, self.config['feature_num'], self.config['feature_dim'])

        dnn_output = self.dnn(dnn_input)
        cin_output = self.cin(cin_input)
        lin_output = self.lin(lin_input)

        output = self.sigm(self.w_dnn(dnn_output) + self.w_cin(cin_output) + self.w_lin(lin_output))
        return output
    

    def loss(self, y_pred, y_true):
        return self.loss_f(y_pred, y_true)
