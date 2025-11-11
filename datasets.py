import torch
import torch.utils.data as uitilsData
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from pickle import dump
from pickle import load
from numpy import save
from numpy import load as load_npy


class BenchMarkDataset(uitilsData.Dataset):
    """
    Prepare the dataset or a gauge for the models
    
    Parameters:
        path, str:
            Path to dataset which contains train and test set for each gauge as csv files
        sensorID, int:
            id of prepared gauge
        split, str:
            identify whether train or test set will be prepared
        scaler, obj:
            minmax scaler which is created based on train set
    
    """

    def __init__(self, sensorID=668, split="train", path='../streamflow-stations/'):

        self.path = path
        self.split = split
        self.sensorID = sensorID
        
        self.read()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx,:,:]
        y = self.y[idx,:]
        
        return X, y
    
    def get_values(self):
        
        return (self.scalerX, self.scalerY)

    def read_file(self, file_path):
        X_path = file_path + "_x.npy"
        y_path = file_path + "_y.npy"
        with open( X_path, 'rb') as f:
             X = np.load(f)
        with open( y_path, 'rb') as f:
             y = np.load(f)

        x_train_history = X[:,:72*3].reshape(-1, 72, 3)
        x_train_future = X[:,72*3:].reshape(-1, 24, 2)
        x_train_future = x_train_future[:,:,[0,1,1]]
        last = x_train_history[:, -1, 2]
        last = np.tile(last.reshape(last.shape[0], 1), (1, 24))
        x_train_future[:,:,2] = last
        ds_X = np.concatenate([x_train_history,x_train_future],axis=1)

        return ds_X, y

    def read(self):


    
        DATASET_PATH = self.path
        STATION_ID = self.sensorID

        self.scalerX = load(open( '{path}{station_id}_npy_24/{station_id}_scaler_x.pkl'.format(path=DATASET_PATH, station_id=STATION_ID), 'rb'))
        self.scalerY = load(open( '{path}{station_id}_npy_24/{station_id}_scaler_y.pkl'.format(path=DATASET_PATH, station_id=STATION_ID), 'rb'))
        
        file_path = "{path}{station_id}_npy_24/{station_id}_{split}".format(
            path=self.path, station_id=self.sensorID, split=self.split)
        X, y = self.read_file(file_path)


        ds_X = torch.Tensor(X)
        ds_y = torch.Tensor(y)
        self.X = ds_X
        self.X = self.X.permute(0, 1, 2)
        self.y = ds_y


class RainDataset(torch.utils.data.Dataset):
    def __init__(self, type, path='../iowaRain/Version#1/training_validation_test_tensors.pt'):
        self.type = type
        self.path = path
        self.read()
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        (f, l) = self.X[idx]
        i = self.y[idx]
        combine = torch.cat((f, l), 0)
        #combine = torch.squeeze(combine, 0)
        #i = torch.squeeze(i, 0)
        # return (f, l), i
        return combine, i

    def read(self):
        loaded_data = torch.load(self.path, weights_only=False)
        self.X = loaded_data[f'x_{self.type}']
        self.y = loaded_data[f'y_{self.type}']
        print(len(self.X))
