import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader
import pickle




class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]


        
        return (x, y)
    
    def __len__(self):
        return len(self.labels)

def load_pdbbind(data_dir='data/pdbbind_data.pickle'):


    with open(data_dir,'rb') as f:
        all_datasets = pickle.load(f)
    print(all_datasets.keys())

    data = all_datasets['PDBbind']

    X_mean = data['train']['X'].mean()
    y_mean = data['train']['y'].mean() 
    X_std  = data['train']['X'].std()
    y_std = data['train']['y'].std()  

    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']


    x_train = torch.tensor((train_data['X']-X_mean)/X_std, dtype=torch.float32)
    y_train = torch.tensor((train_data['y']-y_mean)/y_std, dtype=torch.float32)
    x_valid = torch.tensor((valid_data['X']-X_mean)/X_std, dtype=torch.float32)
    y_valid = torch.tensor((valid_data['y']-y_mean)/y_std, dtype=torch.float32)
    x_test = torch.tensor((test_data['X']-X_mean)/X_std, dtype=torch.float32)
    y_test = torch.tensor((test_data['y']-y_mean)/y_std, dtype=torch.float32)
    # x_valid=torch.tensor(valid_data['X'], dtype=torch.float32)
    # y_valid=torch.tensor(valid_data['y'], dtype=torch.int64)


    train_set = MyDataset(x_train, y_train)
    valid_set = MyDataset(x_valid, y_valid)
    test_set = MyDataset(x_test, y_test)


    n_features = 2052 # 784
    n_classes = 1
    print('-----')
    print('y standard deviation ', y_std)
    print('-----')

    return train_set, valid_set, test_set, n_features, n_classes


