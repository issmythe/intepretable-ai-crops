import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr

from performance_tracker import PerformanceTracker

from networks.hybrid_lstm import HybridLSTM


###############################
##### Helpers to get data #####
###############################


def split_train_test(df, test_years, val_years=None, train_years=None):
    np.random.seed(123)
    val_years = val_years if val_years is not None else []
    
    # train_pd = df[~df['year'].isin(list(test_years) + list(val_years))].sample(frac=1)
    # if train_years:
    #     train_pd = train_pd[train_pd['year'].isin(train_years)]
    
    if train_years:
        train_pd = df[df['year'].isin(train_years)].sample(frac=1)
    else:
        train_pd = df[~df['year'].isin(list(test_years) + list(val_years))]
    val_pd = df[df['year'].isin(val_years)]
    test_pd = df[df['year'].isin(test_years)]

    return train_pd, val_pd, test_pd


def get_model_data(df, ss_x, ss_x_flat, ss_y, ts_cols, flat_cols, train, batch_size=64,
                   norm_x=True, reshape_dims=None, y='log_yield'):
    if norm_x:
        x = []
        for i in range(len(ts_cols)):
            xrow = df[ts_cols[i]].to_list()
            x.append(ss_x[i].fit_transform(xrow) if train else ss_x[i].transform(xrow))
        x = np.stack(x, axis=1)
    else:
        x = np.stack([np.array(df[c].to_list()) for c in ts_cols], axis=1)

    n, vars, days = x.shape
    x = x.reshape(n, 1, vars, days)
    if reshape_dims:
        period_len, n_periods = reshape_dims
        x = x[:,:,:,:period_len * n_periods]
        x = x.reshape([x.shape[0], x.shape[1], x.shape[2], period_len, n_periods])

    x_flat = np.array(df[flat_cols])
    x_flat = ss_x_flat.fit_transform(x_flat) if train else ss_x_flat.transform(x_flat)

    y_sz = len(df[y].iloc[0]) if isinstance(df[y].iloc[0], list) else 1
    y = np.stack(df[y]).reshape(n, y_sz)
    y = ss_y.fit_transform(y) if train else ss_y.transform(y)

    ds = TensorDataset(torch.Tensor(x).type(torch.float32),
                       torch.Tensor(x_flat).type(torch.float32),
                       torch.Tensor(y).type(torch.float32))
    loader = DataLoader(ds, batch_size=batch_size)
    return (x, x_flat), y, ds, loader


######################
##### Train/test #####
######################
def train(model, device, train_loader, optimizer, criterion):
    running_loss = 0.0

    model.train()
    for batch_idx, (x, x_flat, target) in enumerate(train_loader):
        inputs, labels = (x.to(device), x_flat.to(device)), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return (running_loss / (batch_idx + 1)) ** 0.5


def test(model, device, test_x, test_y, criterion, predict=False):
    model.eval()
    with torch.no_grad():
        x, x_flat = test_x
        data = (torch.Tensor(x).to(device), torch.Tensor(x_flat).to(device))
        target = torch.Tensor(test_y).to(device)
        outputs = model(data)
        detached = outputs.detach().cpu().numpy() \
            if (device.type == 'cuda' or device.type == 'mps') \
            else outputs.detach().numpy()
        if outputs.shape[-1] == 1:
            detached = detached.flatten()
        if predict:
            return detached
        # print('*', outputs.shape, target.shape)
        mse = criterion(outputs, target).item()
        corr = pearsonr(detached.flatten(), test_y.flatten())[0]
        

    return mse ** 0.5, corr


#############################################
##### Fit and predict one fold of years #####
#############################################

def fit_predict_fold(data_obj, ts_cols, flat_cols, model, optimizer, max_epochs,
                     device, criterion, tol=None, verbose=False, print_train=False):
    # Setup vars
    pt = PerformanceTracker(max_epochs, verbose)
    pt_train = PerformanceTracker(max_epochs, verbose)

    val_rmse, val_corr = test(model, device, data_obj.val_x, data_obj.val_y, criterion)
    pt.update(val_rmse, val_corr)

    # Train
    t0 = time.time()
    for epoch in range(max_epochs):
        train_rmse = train(model, device, data_obj.train_loader, optimizer, criterion)
        val_rmse, val_corr = test(model, device, data_obj.val_x, data_obj.val_y, criterion)
        pt.update(val_rmse, val_corr)
        if print_train:
            train_rmse, train_corr = test(
                model, device, data_obj.train_x, data_obj.train_y, criterion)
            pt_train.update(train_rmse, train_corr)

    # Predict
    preds = test(model, device, data_obj.test_x, data_obj.test_y, criterion, predict=True)
    pred_df = data_obj.test_pd[['year', 'fips', 'log_yield']].assign(
        pred=data_obj.ss_y.inverse_transform(preds.reshape(data_obj.test_y.shape)).tolist()) #.flatten())
    if data_obj.test_y.shape[1] == 1:
        pred_df['pred'] = pred_df['pred'].apply(lambda x: x[0])
    return pred_df, pt.get_best()


############################
##### Pretrain helpers #####
############################

def set_gradients(model, freeze_layers, unfreeze):
    for name, param in model.named_parameters():
            if name in freeze_layers:
                param.requires_grad = unfreeze
            else:
                break

def get_freeze_layers(pretrain_loc):
    freeze_layers = ['conv2d.weight', 'conv2d.bias']
    if pretrain_loc == 1:
        freeze_layers += ['fc_pretrain.weight', 'fc_pretrain.bias']
    if pretrain_loc >= 2:
        freeze_layers += ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0']
    if pretrain_loc == 3:
        freeze_layers += ['fc1.weight', 'fc1.bias']
    return freeze_layers


def fit_predict_pretrain_fold(pretrain_do, full_do, pretrain_loc, epochs, lrs, device,
                              criterion, ts_cols, flat_cols, verbose=False, return_model=False):
    
    model = HybridLSTM(use_best_params=True, use_ddays=False, read_params=False,
                       verbose=False, pretrain_loc=pretrain_loc, pretrain=True)
    model.to(device)
    freeze_layers = get_freeze_layers(pretrain_loc)
    n_epochs_pt1, n_epochs_pt2, n_epochs_combined = epochs
    lr1, lr2, lr_combo = lrs

    # Train first half
    optimizer = optim.Adam(model.parameters(), lr=lr1)
    preds, _ = fit_predict_fold(pretrain_do, ts_cols, flat_cols, model, optimizer, n_epochs_pt1, device,
                        criterion, verbose=verbose, print_train=False)

    # Freeze first half + train second half
    set_gradients(model, freeze_layers, False)
    model.pretrain = False
    optimizer = optim.Adam(model.parameters(), lr=lr2)
    preds, _ = fit_predict_fold(full_do, ts_cols, flat_cols, model, optimizer, n_epochs_pt2, device,
                        criterion, verbose=verbose, print_train=False)

    # Train jointly & predict
    set_gradients(model, freeze_layers, True)
    optimizer = optim.Adam(model.parameters(), lr=lr_combo)
    preds, _ = fit_predict_fold(full_do, ts_cols, flat_cols, model, optimizer, n_epochs_combined,
                                device, criterion, verbose=verbose, print_train=False)
    if return_model:
        return preds, model
    else:
        return preds













