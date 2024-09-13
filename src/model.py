import numpy as np
import torch
import sklearn

def get_n_blocks(len_series: int, filter_size: int, base_dilation: int = 2) -> int:
    '''
    Function to determine the minimum number of residual blocks for full coverage of the input time series.
    
    Args:
        len_series: Length of the time series.
        filter_size: Filter size of the 1D convolutions of the TCN.
        base_dilation: Base of the dilation; for the i-th block of the TCN, it is supposed to be `base_dilation`**i.
        
    Returns:
        n_blocks: Minimum number of residual blocks for having full coverage of the input time series.
    '''
    if base_dilation == 2:
        log = np.log2(1 + (len_series - 1)/(2*(filter_size - 1)))
    else:
        log = np.log(1 + (len_series - 1)/(2*(filter_size - 1)))
        log /= np.log(base_dilation)
    #
    n_blocks = np.ceil(log)
    return int(n_blocks)

class ResidualBlock(torch.nn.Module):
    def __init__(self, num_chan: int, dilation: int, dict_params: dict, last_block: bool,
                 gated_activation: bool = False) -> None:
        '''
        Residual block of the TCN.

        Args:
            num_chan: Number of features of the input time series. For a hidden layer, this is the number of filters of the previous one.
            dilation: Dilation factor.
            dict_params: Dictionary containing information about the model architecture.
            last_block: Whether it is the last residual block of the TCN.
            gated_activation: Whether to use gated (i.e., tanh*sigmoid) activation function; if false, relu is used.

        Returns: None.
        '''
        super().__init__()
        #
        filter_size = dict_params['filter_size']
        frac_dropout = dict_params['frac_dropout']
        num_filters = dict_params['num_filters']
        #
        self.padding = (filter_size - 1)*dilation
        self.last_block = last_block
        self.gated_activation = gated_activation
        # first convolution
        self.conv_1 = torch.nn.Conv1d(in_channels = num_chan, out_channels = num_filters, kernel_size = filter_size,
                                dilation = dilation)
        self.conv_1 = torch.nn.utils.parametrizations.weight_norm(self.conv_1)
        # second convolution
        self.conv_2 = torch.nn.Conv1d(in_channels = num_filters, out_channels = num_filters, kernel_size = filter_size,
                                dilation = dilation)
        self.conv_2 = torch.nn.utils.parametrizations.weight_norm(self.conv_2)
        # 1D convolution
        self.conv_1x1 = torch.nn.Conv1d(in_channels = num_chan, out_channels = num_filters, kernel_size = 1)
        #
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p = frac_dropout)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.conv_1(torch.nn.functional.pad(x, (self.padding, 0)))
        if self.gated_activation == False:
            y = self.relu(y)
        else:
            y = self.tanh(y)*self.sigmoid(y)
        y = self.dropout(y)
        #
        y = self.conv_2(torch.nn.functional.pad(y, (self.padding, 0)))
        if self.last_block == False:
            if self.gated_activation == False:
                y = self.relu(y)
            else:
                y = self.tanh(y)*self.sigmoid(y)
        y = self.dropout(y)
        #
        return self.conv_1x1(x) + y

class TCN(torch.nn.Module):
    def __init__(self, len_series: int, num_feat: int, len_output: int, dict_params: dict, gated_activation: bool = False) -> None:
        '''
        TCN architecture.

        Args:
            len_series: Length of the input series.
            num_feat: Number of features of the input series.
            len_output: Length of the output series.
            dict_params: Dictionary containing information about the model architecture.
            gated_activation: Whether to use gated (i.e., tanh*sigmoid) activation function; if false, relu is used.

        Returns: None.
        '''
        super().__init__()
        #
        dict_params = dict_params['model']
        #
        filter_size = dict_params['filter_size']
        frac_dropout = dict_params['frac_dropout']
        base_dilation = dict_params['base_dilation']
        num_filters = dict_params['num_filters']
        self.len_output = len_output
        # get number of blocks
        n_blocks = get_n_blocks(len_series = len_series, filter_size = filter_size, base_dilation = base_dilation)
        # build TCN
        list_blocks = []
        for i in range(n_blocks):
            if i == 0:
                list_blocks.append(ResidualBlock(num_chan = num_feat, dilation = base_dilation**i, dict_params = dict_params,
                                                 last_block = False, gated_activation = gated_activation))
            elif 0 < i < n_blocks - 1:
                list_blocks.append(ResidualBlock(num_chan = num_filters, dilation = base_dilation**i, dict_params = dict_params,
                                                 last_block = False, gated_activation = gated_activation))
            else:
                list_blocks.append(ResidualBlock(num_chan = num_filters, dilation = base_dilation**i, dict_params = dict_params,
                                                 last_block = True, gated_activation = gated_activation))
        self.tcn = torch.nn.ModuleList(list_blocks)
        # final convolutional layer, used to fix dimensions
        self.conv_final = torch.nn.Conv1d(in_channels = num_filters, out_channels = 1, kernel_size = 1)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # TCN
        y = self.tcn[0](x)
        if len(self.tcn) > 0:
            for i in range(1, len(self.tcn)):
                y = self.tcn[i](y)
        # final convolution
        y = self.conv_final(y[:, :, -self.len_output:])
        return y
    
class TrainTCN():
    def __init__(self, model: torch.torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader) -> None:
        '''
        Class to train the TCN model.
        
        Args:
            model: PyTorch model.
            dict_params: Dictionary containing information about the training strategy.
            dataloader_train: Dataloader containing training data.
            dataloader_valid: Dataloader containing validation data.
            
        Returns: None.
        '''
        self.model = model
        self.dict_params = dict_params
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        #
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, mode = 'min', factor = 0.5, patience = 10, threshold = 1e-4,
                                                                    threshold_mode = 'rel', verbose = False)

    def _model_on_batch(self, batch: tuple, model: torch.torch.nn.Module, optimizer: torch.optim, loss_func: torch.torch.nn.modules.loss,
                        perform_training: bool = True) -> float:
        '''
        Function to perform training on a single batch of data.
        
        Args:
            batch: Batch of data to use for training/evaluation.
            model: PyTorch model.
            optimizer: Optimizer to use.
            loss_func: Loss function to use.
            perform_training: Whether to perform training (if not, evaluation is understood).
            
        Returns:
            loss: Value of the loss function.
        '''
        if perform_training == True:
            optimizer.zero_grad()
        # get data from the batch
        x, y_true = batch
        x = x.to('cpu')
        y_true = y_true.to('cpu')
        # make predictions
        y_hat = model(x).to('cpu')
        # compute the loss function
        loss = loss_func(y_true, y_hat)
        if perform_training == True:
            loss.backward()
            optimizer.step()
        #
        return loss.item()

    def _train(self) -> float:
        '''
        Function to train the TCN model on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the training loss function per batch.
        '''
        model = self.model
        optimizer = self.optimizer
        loss_func = self.loss_func
        loader = self.dataloader_train
        #
        model.train()
        loss_epoch = 0
        # iterate over batches
        for batch in loader:
            loss_epoch += self._model_on_batch(batch = batch, model = model, optimizer = optimizer, loss_func = loss_func,
                                               perform_training = True)
        #
        return loss_epoch/len(loader)
    
    def _eval(self) -> float:
        '''
        Function to evaluate the TCN model on the validation set on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the validation loss function per batch.
        '''
        model = self.model
        loss_func = self.loss_func
        loader = self.dataloader_valid
        #
        model.eval()
        loss_epoch = 0
        # iterate over batches
        with torch.no_grad():
            for batch in loader:
                loss_epoch += self._model_on_batch(batch = batch, model = model, optimizer = None, loss_func = loss_func,
                                                   perform_training = False)
        #
        return loss_epoch/len(loader)
    
    def train_model(self) -> (torch.torch.nn.Module, list, list):
        '''
        Function to train the TCN model.
        
        Args: None.
            
        Returns:
            model: Trained TCN model.
            list_loss_train: List of training loss function across the epochs.
            list_loss_valid: List of validation loss function across the epochs.
        '''
        model = self.model
        dict_params = self.dict_params
        n_epochs = dict_params['training']['n_epochs']
        patience = dict_params['training']['patience']
        min_improve_valid_loss = dict_params['training']['min_improve_valid_loss']
        #
        list_loss_train, list_loss_valid = [], []
        counter_patience = 0
        for epoch in range(1, n_epochs + 1):
            loss_train = self._train()
            loss_valid = self._eval()
            # check validation loss improvement for patience
            if (len(list_loss_valid) > 0) and (loss_valid >= np.nanmin(list_loss_valid)*(1 - min_improve_valid_loss)):
                counter_patience += 1
            # check validation loss w.r.t. best value
            if (len(list_loss_valid) == 0) or (loss_valid < np.nanmin(list_loss_valid)):
                torch.save(self.model.state_dict(), '../data/artifacts/weights.p')
                counter_patience = 0
            # scheduler step
            self.scheduler.step(loss_valid)
            #
            print(f'Epoch {epoch}: training loss = {loss_train:.4f}, validation loss = {loss_valid:.4f}. ' +
                  f'Learning rate = {self.optimizer.param_groups[0]["lr"]}. Patience = {counter_patience}')
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
            # stop training with patience
            if counter_patience >= patience:
                print(f'Training stopped at epoch {epoch}; restoring weights from epoch {np.argmin(list_loss_valid) + 1}')
                self.model.load_state_dict(torch.load('../data/artifacts/weights.p'))
                break
        #
        return model, list_loss_train, list_loss_valid
    
def get_y_true_y_hat(model: torch.nn.Module, x: torch.tensor, y: torch.tensor, date_y: np.array,
                     scaler: sklearn.preprocessing.StandardScaler) -> (np.array, np.array):
    '''
    Function to get the real time series and its prediction.
    
    Args:
        model: Trained TCN model.
        x: Tensor representing regressors.
        y: Tensor representing target time series.
        date_y: Array containing the dates corresponding to the elements of `y`.
        scaler: Scaled used to rescale data.
        
    Returns:
        y_true: Array containing the true values.
        y_hat: Array containing the predicted values.
    '''
    list_date = []
    y_true = []
    y_hat = []
    preds = model(x)
    for i in range(np.unique(date_y).shape[0]):
        date = np.unique(date_y)[i]
        list_date.append(date)
        idx = np.where(date_y == date)
        y_true.append(y.numpy()[idx].mean())
        y_hat.append(preds.detach().numpy()[idx].mean())
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    # scale back
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_hat = scaler.inverse_transform(y_hat.reshape(-1, 1)).ravel()
    #
    return y_true, y_hat
  
def compute_mape(y_true: np.array, y_hat: np.array) -> float:
    '''
    Function to compute the MAPE.
    
    Args:
        y_true: Array containing the true values.
        y_hat: Array containing the predicted values.
        
    Returns:
        mape: MAPE computed from `y_true` and `y_hat`.
    '''
    mape = np.mean(abs(y_true[y_true > 0] - y_hat[y_true > 0])/y_true[y_true > 0])
    return mape