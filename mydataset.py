from __future__ import print_function
import os
import sys
import random
import torch
import torch.utils.data as data
from collections import defaultdict
import numpy as np
from utils import data_preprocess

class ListDataset(data.Dataset):
    def __init__(self, list_file, category_emb, field_size):        
        result_dict = data_preprocess.read_criteo_data(list_file, category_emb)

        self.feature_size = result_dict['feature_sizes']
        Xi = result_dict['index']
        Xv = result_dict['value']
        y = result_dict['label']        
        self.length = len(y)

        Xi = np.array(Xi).reshape((-1, field_size, 1))
        Xv = np.array(Xv)
        y = np.array(y)
        
        self.Xi = torch.LongTensor(Xi)
        self.Xv = torch.FloatTensor(Xv)
        self.Y = torch.FloatTensor(y)
        print('dataset size: ', self.Xi.shape, self.Xv.shape, self.Y.shape)
    def __getitem__(self, idx):
        xi = self.Xi[idx]
        xv = self.Xv[idx]
        y = self.Y[idx]
        return xi, xv, y

    def __len__(self):
        return self.length