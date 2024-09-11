The configuration file is organized as follows:

- _data_:
    - _len_series_: length of the input series;
    - _size_train_: fraction of data used for training;
    - _size_valid_: fraction of data used for validation;
    - _horizon_forecast_: forecasting horizon (length of the output series);
- _model_:
    - _filter_size_: filter size of the CNN;
    - _frac_dropout_: fraction of dropout;
    - _base_dilation_: base of the dilation; the dilation of the $i$-th residual block is $b^i$, where $b$ is `base_dilation`;
    - _num_filters_: number of filters in the CNN:
- _training_:
    - _batch_size_: batch size used for training;
    - _n_epochs_: number of epochs used for training;
    - _patience_: patience used for early stopping;
    - _min_improve_valid_loss_: minimum required improvement of the validation loss; if its change is smaller than this value, then the patience counter increases.