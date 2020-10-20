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

def load_bace_HIV(data_name = 'bace', data_dir = 'data/iclr_data.pickle'):


    with open(data_dir,'rb') as f:
        all_datasets = pickle.load(f)

    data = all_datasets['base_split'] if data_name == 'bace' else all_datasets['HIV_split']

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


