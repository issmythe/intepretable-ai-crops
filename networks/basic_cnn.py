import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network import Network


class BasicCNN(Network):
    def __init__(self, use_best_params, kwargs=None, use_ddays=True, x_dim=None,
                 x_flat_dim=None, read_params=False, verbose=False):
        super(BasicCNN, self).__init__()
        self.network_name = 'basic_cnn'
        self.init_params(
            use_best_params, kwargs, use_ddays, x_dim, x_flat_dim, read_params, verbose)

        x_dim = x_dim if x_dim else self.DEFAULT_X_DIM
        x_flat_dim = x_flat_dim if x_flat_dim else self.DEFAULT_X_FLAT_DIM

        self.conv2d = nn.Conv2d(1, self.best_kwargs['cdim1'], (3, self.best_kwargs['k']),
                                self.best_kwargs['stride'])
        c1_dim = self.conv_calc(x_dim, 0, self.best_kwargs['stride'], self.best_kwargs['k'])

        self.conv1d = nn.Conv1d(self.best_kwargs['cdim1'], self.best_kwargs['cdim1'],
                                self.best_kwargs['k'], self.best_kwargs['stride'])
        c2_dim = self.conv_calc(int(c1_dim / 2), 0, self.best_kwargs['stride'],
                                self.best_kwargs['k'])

        self.dropout1 = nn.Dropout2d(self.best_kwargs['dropout'])
        self.dropout2 = nn.Dropout1d(self.best_kwargs['dropout'])
        self.dropout3 = nn.Dropout1d(self.best_kwargs['dropout'])

        flattened_dim = int((c2_dim * self.best_kwargs['cdim1']) / 2)
        self.cnn_fc = nn.Linear(flattened_dim, self.best_kwargs['ldim'])
        self.fc1 = nn.Linear(x_flat_dim + self.best_kwargs['ldim'], self.best_kwargs['ldim'])
        self.fc2 = nn.Linear(self.best_kwargs['ldim'], 1)


    def __init_params(self, use_best_params, kwargs, use_ddays, read_params, verbose):
        self.verbose = verbose

        self.best_kwargs = dict()
        if use_best_params:
            if read_params:
                self.best_kwargs = self.get_best_params(use_ddays)
            else:
                self.best_kwargs = self.get_best_params_hardcoded(use_ddays)
        if kwargs is not None:
            self.best_kwargs.update(kwargs)


    def get_best_params_hardcoded(self, use_ddays):
        if use_ddays:
            return {'cdim1': 8, 'ldim': 32, 'dropout': 0.1, 'lr': 0.0001, 'k': 7, 'stride': 4,
                    'n_epochs': 100, 'rmse': 0.212, 'corr': 0.782, 'filtered_len': 26331,
                    'rmse_f': 0.212, 'corr_f': 0.782, 'best_iters': '[92, 20, 98, 92, 32]',
                    'param_combo': 30}
        else:
            return {'cdim1': 8, 'ldim': 32, 'dropout': 0.1, 'lr': 7.5e-05, 'k': 5, 'stride': 2,
                    'n_epochs': 100, 'rmse': 0.224, 'corr': 0.756, 'filtered_len': 25984,
                    'rmse_f': 0.224, 'corr_f': 0.756, 'best_iters': '[30, 38, 90, 86, 99]',
                    'param_combo': 28}


    def get_param_grid(self):
        return {'cdim1': [4, 8, 16],
                'ldim': [8, 16, 32],
                'dropout': [0, 0.1, 0.2],
                'lr': [1e-4, 5e-5, 7.5e-5],
                'k': [5, 7, 9],
                'stride': [2, 4, 6],
                'n_epochs': [25, 50, 100, 150]}


    def forward(self, x):
        x, x_flat = x
        i = self.quick_print(x_flat, 0)
        i = self.quick_print(x, 0)

        # Convolution #1
        x = self.conv2d(x)
        x = F.relu(x)
        x = self.dropout1(x)
        i = self.quick_print(x, i) #1

        x = torch.flatten(x, 2)
        i = self.quick_print(x, i) #2

        # Average pool
        x = F.avg_pool1d(x, 2)
        i = self.quick_print(x, i) #3

        # Convolution #2
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.dropout2(x)
        i = self.quick_print(x, i) #4

        x = torch.flatten(x, 1)
        i = self.quick_print(x, i) #5

        # Average pool
        x = F.avg_pool1d(x, 2)
        i = self.quick_print(x, i) #6

        # CNN FC layer
        # x = self.dropout3(x)
        x = self.cnn_fc(x)
        x = F.relu(x)
        i = self.quick_print(x, i) #7

        # FC all features
        x = torch.cat((x, x_flat), 1)
        i = self.quick_print(x, i) #8
        x = self.fc1(x)
        x = F.relu(x)

        return self.fc2(x)
