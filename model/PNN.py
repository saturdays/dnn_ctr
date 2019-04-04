# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of PNN

Reference:
[1] Product-based Neural Networks for User Response Prediction
Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu Shanghai Jiao Tong University
{kevinqu, hcai, kren, wnzhang, yyu}@apex.sjtu.edu.cn Ying Wen, Jun Wang University College London {ying.wen, j.wang}@cs.ucl.ac.uk

"""
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn

"""
    网络结构部分
"""

class PNN(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    use_inner_product: use inner product or not?
    use_outer_product: use outter product or not?
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=950104 someone's birthday, my lukcy number
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?


    Attention: only support logsitcs regression
    """

    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth = 3, deep_layers=[32, 32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5], use_inner_product = True, use_outer_product = False,
                 deep_layers_activation='relu', n_epochs=64, batch_size=256, learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, verbose=False, random_seed=950104,weight_decay=0.0, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1, greater_is_better=True
                 ):
        super(PNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use inner_product or outer_product
        """
        if self.use_inner_product and self.use_inner_product:
            print("The model uses both inner product and outer product")
        elif self.use_inner_product:
            print("The model uses inner product (IPNN))")
        elif self.use_ffm:
            print("The model uses outer product (OPNN)")
        else:
            print("The model is sample deep model only! Neither inner product or outer product is used")

        """
            embbedding part
        """
        print("Init embeddings")
        self.embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        print("Init embeddings finished")

        """
            first order part (linear part)
        """
        print("Init first order part")
        self.first_order_weight = nn.ModuleList([nn.ParameterList([torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)]) for i in range(self.deep_layers[0])])
        self.bias = torch.nn.Parameter(torch.randn(self.deep_layers[0]), requires_grad=True)
        print("Init first order part finished")

        """
            second order part (quadratic part)
        """
        print("Init second order part")
        if self.use_inner_product:
            self.inner_second_weight_emb = nn.ModuleList([nn.ParameterList([torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)]) for i in range(self.deep_layers[0])])

        if self.use_outer_product:
            arr = []
            for i in range(self.deep_layers[0]):
                tmp = torch.randn(self.embedding_size,self.embedding_size)
                arr.append(torch.nn.Parameter(torch.mm(tmp,tmp.t())))
            self.outer_second_weight_emb = nn.ParameterList(arr)
        print("Init second order part finished")


        print("Init nn part")

        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i) + '_dropout', nn.Dropout(self.dropout_deep[i]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)
        print("Init nn part succeed")

        print "Init succeed"

    def forward(self, Xi, Xv):
        """
        :param Xi: index input tensor, batch_size * k * 1
        :param Xv: value input tensor, batch_size * k * 1
        :param is_pretrain: the para to decide fm pretrain or not
        :return: the last output
        """

        """
            embedding
        """
        emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.embeddings)]

        """
            first order part (linear part)
        """
        first_order_arr = []
        for i, weight_arr in enumerate(self.first_order_weight):
            tmp_arr = []
            for j, weight in enumerate(weight_arr):
                tmp_arr.append(torch.sum(emb_arr[j]*weight,1))
            first_order_arr.append(sum(tmp_arr).view([-1,1]))
        first_order = torch.cat(first_order_arr,1)

        """
            second order part (quadratic part)
        """
        if self.use_inner_product:
            inner_product_arr = []
            for i, weight_arr in enumerate(self.inner_second_weight_emb):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_*sum_).view([-1,1]))
            inner_product = torch.cat(inner_product_arr,1)
            first_order = first_order + inner_product

        if self.use_outer_product:
            outer_product_arr = []
            emb_arr_sum = sum(emb_arr)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1,self.embedding_size,1]),emb_arr_sum.view([-1,1,self.embedding_size]))
            for i, weight in enumerate(self.outer_second_weight_emb):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr*weight,2),1).view([-1,1]))
            outer_product = torch.cat(outer_product_arr,1)
            first_order = first_order + outer_product

        """
            nn part
        """
        if self.deep_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu
        x_deep = first_order
        for i, h in enumerate(self.deep_layers[1:], 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i) + '_dropout')(x_deep)
        x_deep = self.deep_last_layer(x_deep)
        return torch.sum(x_deep, 1)