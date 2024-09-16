import os

import numpy as np
import pandas as pd
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.DL_DATA_PATH = 'drive/MyDrive/current_research_projects/dl_yield_forecasts/data/'
        self.DEFAULT_X_DIM = 274
        self.DEFAULT_X_FLAT_DIM = 1923


    def init_params(self, use_best_params, kwargs, use_ddays,
                    x_dim, x_flat_dim, read_params, verbose):
        self.verbose = verbose
        
        self.best_kwargs = dict()
        if use_best_params:
            if read_params:
                self.best_kwargs = self.get_best_params(use_ddays)
            else:
                self.best_kwargs = self.get_best_params_hardcoded(use_ddays)
        if kwargs is not None:
            self.best_kwargs.update(kwargs)


    def __read_results(self, use_ddays):
        gs_dir = f'{self.DL_DATA_PATH}/grid_search/{self.network_name}_dday' if use_ddays else \
            f'{self.DL_DATA_PATH}/grid_search/{self.network_name}_temperature'
        try:
            return pd.read_csv(f'{gs_dir}.csv')
        except FileNotFoundError:
            return pd.concat([pd.read_csv(f'{gs_dir}/{f}') for f in os.listdir(gs_dir)])\
                .drop_duplicates(subset='param_combo')


    def get_best_params(self, use_ddays, allow_incomplete=False):
        results = self.__read_results(use_ddays)
        
        if results['param_combo'].nunique() != 100:
            if allow_incomplete:
                print(f"Warning: {results['param_combo'].nunique()} / 100 runs found")
            else:
                raise AssertionError(
                    f"Error: {results['param_combo'].nunique()} / 100 runs found")

        best_params = results[results['rmse'] == results['rmse'].min()]
        return {k: v[0] for k, v in best_params.to_dict(orient='list').items()}


    def conv_calc(self, in_dim, pad, stride, k):
        out = int(np.floor((in_dim + 2 * pad - (k - 1) - 1) / stride + 1))
        return out


    def quick_print(self, x, i):
        if self.verbose:
            print(i, x.shape)
        return i + 1


    def get_best_param(self, arg):
        return self.best_kwargs[arg]
