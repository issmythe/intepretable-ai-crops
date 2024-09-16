import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network import Network

class BaselineNN(Network):
    def __init__(self, use_best_params, kwargs=None, use_ddays=True, x_dim=None,
                 x_flat_dim=None, n_ts_cols=3, read_params=False, verbose=False):
        super(BaselineNN, self).__init__()

        self.network_name = 'baseline_nn'
        self.init_params(
            use_best_params, kwargs, use_ddays, x_dim, x_flat_dim, read_params, verbose)

        self.x_dim = x_dim if x_dim else self.DEFAULT_X_DIM
        self.x_flat_dim = x_flat_dim if x_flat_dim else self.DEFAULT_X_FLAT_DIM

        self.fc2 = nn.Linear(self.x_flat_dim, self.best_kwargs['fc_dim'])
        self.fc3 = nn.Linear(self.best_kwargs['fc_dim'], 1)
    
        self.dropout = nn.Dropout1d(self.best_kwargs['dropout'])


    def get_best_params_hardcoded(self, use_ddays):
        return {'fc_dim': 16, 'dropout': 0.0, 'lr': 1e-4, 'n_epochs': 25}


    def get_param_grid(self):
        return {'ldim': [8, 16, 32],
                'dropout': [0, 0.1, 0.2],
                'lr': [1e-4, 5e-5, 7.5e-5],
                'n_epochs': [25, 50, 100, 150]}


    def forward(self, x):
        x, x_flat = x
        i = self.quick_print(x_flat, 0) #0

        # FC all features
        x = x_flat
        i = self.quick_print(x, i)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        i = self.quick_print(x, i)      #3

        return self.fc3(x)
