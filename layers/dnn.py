import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.dropout_rate = config['dnn_dropout_rate']
        self.dropout = nn.Dropout(self.dropout_rate)
        self.l2_reg = config['l2_reg_dnn']
        if config['dnn_hidden_layers'] == 0:
            raise ValueError("No layers in DNN")
        
        self.layers = [nn.Linear(config['input_dim'], config['dnn_hidden_dim'])]
        self.activation_layers = [nn.Sigmoid()]


        for i in range(config['dnn_hidden_layers'] - 1):
            self.layers.append(nn.Linear(config['dnn_hidden_dim'], config['dnn_hidden_dim']))
            self.activation_layers.append(nn.Sigmoid())

        self.layers.append(nn.Linear(config['dnn_hidden_dim'], config['output_dim']))
        self.activation_layers.append(nn.Sigmoid())
    
    def forward(self, x):
        deep_x = x
        for i in range(len(self.layers)):
            fc = self.layers[i](deep_x)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_x = fc
        return deep_x
