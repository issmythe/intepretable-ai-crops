import pandas as pd

from dl_helpers import get_model_data, split_train_test
from sklearn.preprocessing import StandardScaler

class DataObjBase:
    def __init__(self):
        pass

    def setup_model_data(self, ts_cols, flat_cols, norm_x, reshape_dims=None, y='log_yield'):
        self.ss_x = [StandardScaler() for i in range(len(ts_cols))]
        self.ss_x_flat = StandardScaler()
        self.ss_y = StandardScaler()

        self.train_x, self.train_y, self.train_dataset, self.train_loader = \
            get_model_data(self.train_pd, self.ss_x, self.ss_x_flat, self.ss_y,
            ts_cols, flat_cols, train=True, norm_x=norm_x, reshape_dims=reshape_dims, y=y)
        self.val_x, self.val_y, self.val_dataset, self.val_loader = \
            get_model_data(self.val_pd, self.ss_x, self.ss_x_flat, self.ss_y,
            ts_cols, flat_cols, train=False, norm_x=norm_x, reshape_dims=reshape_dims, y=y)
        self.test_x, self.test_y, self.test_dataset, self.test_loader = \
            get_model_data(self.test_pd, self.ss_x, self.ss_x_flat, self.ss_y,
            ts_cols, flat_cols, train=False, norm_x=norm_x, reshape_dims=reshape_dims, y=y)


class DataObj(DataObjBase):
    def __init__(self, df, test_years, val_years, train_years, ts_cols, flat_cols, norm_x=True, y='log_yield'):
        self.train_pd, self.val_pd, self.test_pd = \
            split_train_test(df, test_years, val_years, train_years)

        self.setup_model_data(ts_cols, flat_cols, norm_x, y=y)


class DataObj2D(DataObjBase):
    def __init__(self, df, test_years, val_years, train_years, ts_cols, flat_cols, norm_x=True,
                 reshape_dims=(15, 18)):
        self.train_pd, self.val_pd, self.test_pd = \
            split_train_test(df, test_years, val_years, train_years)

        self.setup_model_data(ts_cols, flat_cols, norm_x, reshape_dims)


class DataObjPretrain(DataObjBase):
    def __init__(self, df, test_years, val_years, train_years, ts_cols, flat_cols, norm_x=True,
                 reshape_dims=(15, 18)):
        self.train_pd, self.val_pd, self.test_pd = \
            split_train_test(df, test_years, val_years, train_years)

        self.setup_model_data(ts_cols, flat_cols, norm_x, reshape_dims, y='evi_mini')


class BaggingDataObj(DataObjBase):
    def __init__(self, df, test_years, seed, ts_cols, flat_cols, norm_x=True, y='log_yield'):
        self.test_pd = df[df['year'].isin(test_years)]
        train_df = df[~df['year'].isin(test_years)]
        self.train_pd = train_df.sample(frac=1, replace=True, random_state=seed)
        self.val_pd = pd.concat([self.train_pd, train_df])\
            .drop_duplicates(subset=['fips', 'year'], keep=False)

        self.setup_model_data(ts_cols, flat_cols, norm_x, y=y)


class BaggingDataObj2D(DataObjBase):
    def __init__(self, df, test_years, seed, ts_cols, flat_cols, norm_x=True, reshape_dims=(15, 18)):
        self.test_pd = df[df['year'].isin(test_years)]
        train_df = df[~df['year'].isin(test_years)]
        self.train_pd = train_df.sample(frac=1, replace=True, random_state=seed)
        self.val_pd = pd.concat([self.train_pd, train_df])\
            .drop_duplicates(subset=['fips', 'year'], keep=False)

        self.setup_model_data(ts_cols, flat_cols, norm_x, reshape_dims)


class InterpDataObj(DataObjBase):
    def __init__(self, train_df, ts_cols, flat_cols, y='log_yield'):
        self.ss_x = [StandardScaler() for i in range(len(ts_cols))]
        self.ss_x_flat = StandardScaler()
        self.ss_y = StandardScaler()

        # Fit standard scalars
        _ = get_model_data(train_df, self.ss_x, self.ss_x_flat, self.ss_y,
            ts_cols, flat_cols, train=True, y=y)
        self.ts_cols, self.flat_cols, self.y = ts_cols, flat_cols, y

    def update_test_data(self, test_df):
        self.test_pd = test_df
        self.test_x, self.test_y, self.test_dataset, self.test_loader = \
            get_model_data(self.test_pd, self.ss_x, self.ss_x_flat, self.ss_y,
                           self.ts_cols, self.flat_cols, train=False, y=self.y)


class DummyDataObj(DataObjBase):
    def __init__(self, df, ts_cols, flat_cols):
        train_pd, _, _ = split_train_test(df[:5], [2000], [2000])

        ss_x = [StandardScaler() for i in range(len(ts_cols))]
        ss_x_flat = StandardScaler()
        ss_y = StandardScaler()

        self.x, self.y, _, _ = get_model_data(
            train_pd, ss_x, ss_x_flat, ss_y, ts_cols, flat_cols, train=True, norm_x=True)
