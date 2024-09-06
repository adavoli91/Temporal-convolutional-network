import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

def train_test_splitting(df: pd.DataFrame, dict_params: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    Function to split data in training, validation and test set.

    Args:
        df: Dataframe containing time series. The column representing the main series should be called `y`.
        dict_params: Dictionary containing information about the model architecture.

    Returns:
        df_train: Dataframe corresponding to training set.
        df_valid: Dataframe corresponding to validation set.
        df_test: Dataframe corresponding to test set.
    '''
    df_train = df.iloc[:int(df.shape[0]*dict_params['data']['size_train'])].copy().reset_index(drop = True)
    df_test = df.iloc[int(df.shape[0]*dict_params['data']['size_train']):].reset_index(drop = True).copy().reset_index(drop = True)
    df_valid = df_train.iloc[int(df_train.shape[0]*(1 - dict_params['data']['size_valid'])):].copy().reset_index(drop = True)
    df_train = df_train.iloc[:int(df_train.shape[0]*(1 - dict_params['data']['size_valid']))].copy().reset_index(drop = True)
    # rescale data
    scaler = StandardScaler().fit(df_train[['y']])
    df_train[['y']] = scaler.transform(df_train[['y']])
    df_valid[['y']] = scaler.transform(df_valid[['y']])
    df_test[['y']] = scaler.transform(df_test[['y']])
    #
    return df_train, df_valid, df_test, scaler

def get_x_y(df: pd.DataFrame, df_future: pd.DataFrame, dict_params: dict, test_set: bool = False,
            horizon_forecast: int = None) -> (torch.tensor, torch.tensor, np.array, np.array):
    '''
    Function obtain a tensor of regressors and one of target series.

    Args:
        df: Dataframe containing time series. The column representing the main series should be called `y`.
        df_future: Same as `df`, but corresponding to its future (e.g., `df_valid` could be the "future" of `df_train`).
        dict_params: Dictionary containing information about the model architecture.
        test_set: Whether `df` is the dataframe corresponding to test set.
        horizon_forecast: Length of the series to be predicted.

    Returns:
        x: Tensor representing regressors.
        y: Tensor representing target time series.
        date_x: Array containing the dates corresponding to the elements of `x`.
        date_y: Array containing the dates corresponding to the elements of `y`.
    '''
    dict_params = dict_params['data']
    #
    len_series = dict_params['len_series']
    if (horizon_forecast is None) or (horizon_forecast >= len_series):
        horizon_forecast = len_series
    #
    df_present = df.copy()
    if test_set == False:
        df_future = pd.concat((df_present, df_future)).copy().shift(-len_series)
        df_future = df_future.iloc[:df_present.shape[0]].reset_index(drop = True)
    else:
        df_future = df_present.copy().shift(-len_series).dropna()
        df_present = df_present.iloc[:df_future.shape[0]]
    #
    x = np.array([df_present.loc[i: i + len_series - 1,
                                 [col for col in df_present.columns if col != 'date']].values for i in range(df_present.shape[0] - len_series)])
    y = np.array([df_future.loc[i: i + horizon_forecast - 1, ['y']].values for i in range(df_future.shape[0] - horizon_forecast)])
    date_x = np.array([df_present.loc[i: i + len_series - 1, 'date'].values for i in range(df_present.shape[0] - len_series)])
    date_y = np.array([df_future.loc[i: i + horizon_forecast - 1, 'date'].values for i in range(df_future.shape[0] - horizon_forecast)])
    #
    y = y[:x.shape[0]]
    date_y = date_y[:date_x.shape[0]]
    #
    x = torch.tensor(x.astype(np.float32))
    y = torch.tensor(y.astype(np.float32))
    #
    if len(x.shape) == 2:
        x = x.reshape(x.shape[0], 1, -1)
        y = y.reshape(y.shape[0], 1, -1)
        date_x = date_x.reshape(x.shape[0], 1, -1)
        date_y = date_y.reshape(y.shape[0], 1, -1)
    if len(x.shape) == 3:
        x = torch.tensor(np.einsum('btc -> bct', x))
        y = torch.tensor(np.einsum('btc -> bct', y))
        date_x = date_x.reshape(x.shape[0], 1, -1)
        date_y = date_y.reshape(y.shape[0], 1, -1)
    #
    return x, y, date_x, date_y

class CreateDataset(Dataset):
    def __init__(self, x: torch.tensor, y: torch.tensor):
        '''
        Class to create a PyTorch dataset
        
        Args:
            x: Tensor representing regressors.
            y: Tensor representing target time series.
            
        Returns: None.
        '''
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]
        
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y