import torch
import torch.nn as nn

class CIN(nn.Module):
    def __init__(self, config):
        super(CIN, self).__init__()

        """
        hidden_layers must be a list that contains the dimensions of hidden layers

        For example:
        hidden_layers = [10, 20, 50]
        This means that
        - dimension of X^1 is R^{10 \times D}
        - dimension of X^2 is R^{20 \times D}
        - dimension of X^3 is R^{50 \times D}
        """
        if not isinstance(config['cin_hidden_layers'], list):
            raise ValueError("Wrong type of hidden_layers param. It should be a list.")
        self._hidden_layers = config['cin_hidden_layers']
        
        """
        input_dim is the dimension of input. Should be a tuple of size 2.

        For example: input is X^0 with dimension R^{m \times D}
        Thus, input_dim should be (m, D)
        """
        if not isinstance(config['cin_input_dim'], tuple) or len(config['cin_input_dim']) != 2:
            raise ValueError("Wrong type of input_dim param. It should be a tuple of size 2.")
        self._input_dim = config['cin_input_dim']

        """
        Initianlizing the hidden layers of CIN
        """
        self._W = []
        for i, layer_dim in enumerate(self._hidden_layers):
            H_k1 = self._input_dim[0] if i == 0 else self._hidden_layers[i - 1]
            self._W.append(nn.Parameter(torch.rand(self._input_dim[0], H_k1, layer_dim, requires_grad=True)))
            self.register_parameter(f'cin_weight{i}', self._W[i])

        self._activation_layer = nn.Sigmoid()

    def forward(self, x):
        ps = []
        xs = [x]

        for k, layer_dim in enumerate(self._hidden_layers):
            x_prod = torch.einsum('bil,bjl->bjil', xs[k], xs[0])
            x_next = torch.transpose(torch.einsum('bijk,ijl->bkl', x_prod, self._W[k]), 1, 2)
            ps.append(torch.sum(x_next, 2))
            xs.append(x_next)
        
        p = torch.cat(ps, dim=1)

        return p
