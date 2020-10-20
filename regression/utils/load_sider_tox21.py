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

def load_sider_tox21(data_name='sider', label_idx=0, data_dir_sider='data/sider_data.pickle', data_dir_tox='data/tox_data.pickle'):

    data_dir = data_dir_sider if data_name == 'sider' else data_dir_tox


    with open(data_dir,'rb') as f:
        all_datasets = pickle.load(f)

    data = all_datasets[label_idx]

    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']


    x_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.int64)
    x_valid = torch.tensor(valid_data['X'], dtype=torch.float32)
    y_valid = torch.tensor(valid_data['y'], dtype=torch.int64)
    x_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.int64)
    # x_valid=torch.tensor(valid_data['X'], dtype=torch.float32)
    # y_valid=torch.tensor(valid_data['y'], dtype=torch.int64)


    train_set = MyDataset(x_train, y_train)
    valid_set = MyDataset(x_valid, y_valid)
    test_set = MyDataset(x_test, y_test)


    n_features = 2048 # 784
    n_classes = 2

    return train_set, valid_set, test_set, n_features, n_classes


