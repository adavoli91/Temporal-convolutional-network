import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from preprocessing import *
from model import *

def main():
    dict_params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader)
    # import data
    df = pd.read_csv('../data/raw/train.csv', parse_dates = ['date'], index_col = 'id')
    df = df.groupby(['date', 'family']).agg({'sales': 'sum'}).reset_index()
    # add all dates
    df_temp = []
    for family in df['family'].unique():
        df_temp.append(pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max())}).merge(df[df['family'] == family],
                                                                                                    on = 'date', how = 'left'))
        df_temp[-1]['family'] = family
        df_temp[-1]['sales'] = df_temp[-1]['sales'].ffill()
    df = pd.concat(df_temp).reset_index(drop = True)
    del df_temp
    df = df.rename(columns = {'sales': 'y'})
    #
    df_result = pd.DataFrame()
    #
    for family in ['AUTOMOTIVE']:# df['family'].unique():
        # perform train-test splitting
        df_train, df_valid, df_test, scaler = train_test_splitting(df = df[df['family'] == family].reset_index(drop = True).drop('family', axis = 1),
                                                                dict_params = dict_params)
        #
        horizon_forecast = dict_params['data']['horizon_forecast']
        # training set data
        x_train, y_train, date_x_train, date_y_train = get_x_y(df = df_train, df_future = df_valid, dict_params = dict_params,
                                                            test_set = False, horizon_forecast = horizon_forecast)
        # validation set data
        x_valid, y_valid, date_x_valid, date_y_valid = get_x_y(df = df_valid, df_future = df_test, dict_params = dict_params,
                                                            test_set = False, horizon_forecast = horizon_forecast)
        # test set data
        x_test, y_test, date_x_test, date_y_test = get_x_y(df = df_test, df_future = None, dict_params = dict_params, test_set = True,
                                                        horizon_forecast = horizon_forecast)
        # create datasets and dataloader
        dataset_train = CreateDataset(x = x_train, y = y_train)
        dataset_valid = CreateDataset(x = x_valid, y = y_valid)
        dataloader_train = DataLoader(dataset_train, batch_size = dict_params['training']['batch_size'], shuffle = True)
        dataloader_valid = DataLoader(dataset_valid, batch_size = dict_params['training']['batch_size'], shuffle = False)
        # define the model
        if 'model' in locals():
            del model
        len_series = dataloader_train.dataset.x.shape[2]
        num_feat = dataloader_train.dataset.x.shape[1]
        len_output = dataloader_train.dataset.y.shape[2]
        # model = TCN(len_series = len_series, num_chan = num_chan, dict_params = dict_params, len_input=100, len_output = 7)
        model = TCN(len_series = len_series, num_feat = num_feat, len_output = len_output, dict_params = dict_params,
                    gated_activation = False)
        # perform training
        model, list_loss_train, list_loss_valid = TrainTCN(model = model, dict_params = dict_params,
                                                        dataloader_train = dataloader_train,
                                                        dataloader_valid = dataloader_valid).train_model()
        # load best parameters
        if 'model' in locals():
            del model
        len_series = dataloader_train.dataset.x.shape[2]
        num_feat = dataloader_train.dataset.x.shape[1]
        len_output = dataloader_train.dataset.y.shape[2]
        # model = TCN(len_series = len_series, num_chan = num_chan, dict_params = dict_params, len_input=100, len_output = 7)
        model = TCN(len_series = len_series, num_feat = num_feat, len_output = len_output, dict_params = dict_params,
                    gated_activation = False)
        #
        model.load_state_dict(torch.load('../data/artifacts/weights.p'))
        model.eval()
        # get time series and the corresponding predictions
        y_true_train, y_hat_train = get_y_true_y_hat(model = model, x = x_train, y = y_train, date_y = date_y_train, scaler = scaler)
        y_true_valid, y_hat_valid = get_y_true_y_hat(model = model, x = x_valid, y = y_valid, date_y = date_y_valid, scaler = scaler)
        y_true_test, y_hat_test = get_y_true_y_hat(model = model, x = x_test, y = y_test, date_y = date_y_test, scaler = scaler)
        # compute mape on training, validation and test set
        mape_train = compute_mape(y_true = y_true_train, y_hat = y_hat_train)
        mape_valid = compute_mape(y_true = y_true_valid, y_hat = y_hat_valid)
        mape_test = compute_mape(y_true = y_true_test, y_hat = y_hat_test)
        #
        df_result = pd.concat((df_result, pd.DataFrame({'family': [family], 'mape_train': [mape_train], 'mape_valid': [mape_valid],
                                                        'mape_test': [mape_test]})))
    #
    df_result = df_result.reset_index(drop = True)
    return df_result

if __name__ == '__main__':
    df_result = main()