# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of NFM

Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics
    Xiangnan He,School of Computing,National University of Singapore,Singapore 117417,dcshex@nus.edu.sg
    Tat-Seng Chua,School of Computing,National University of Singapore,Singapore 117417,dcscts@nus.edu.sg

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


"""
    网络结构部分
"""

class DIN(torch.nn.Module):
    def __init__(self,field_size, feature_sizes, embedding_size = 4, is_shallow_dropout = True, dropout_shallow = [0.5],
                 h_depth = 2, deep_layers = [32, 32], is_deep_dropout = True, dropout_deep=[0.0, 0.5, 0.5],
                 deep_layers_activation = 'relu',
                 is_batch_norm = False,
                 use_fm = True, use_ffm = False, 
                 use_high_interaction = True,interation_type = True,
                 ):
        super(DIN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_high_interaction = use_high_interaction
        self.interation_type = interation_type
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        self.pre_train = False

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is nfm(fm+nn layers)")
        elif self.use_ffm:
            print("The model is nffm(ffm+nn layers)")
        else:
            print("You have to choose more than one of (fm, ffm) models to use")
            exit(1)
        """
            bias
        """
        self.bias = torch.nn.Parameter(torch.randn(1))

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            print("Init ffm part succeed")

        """
            high interaction part
        """
        if self.use_high_interaction and self.use_fm:
            self.h_weights = nn.ParameterList([torch.nn.Parameter(torch.ones(self.embedding_size)) for i in range(self.field_size)])
            self.h_bias = nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for i in range(self.field_size)])
            self.h_batch_norm = nn.BatchNorm1d(self.field_size)

        """
            deep part
        """
        print("Init deep part")

        if self.is_deep_dropout:
            self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
        if self.interation_type:
            self.linear_1 = nn.Linear(self.embedding_size, deep_layers[0])
        else:
            self.linear_1 = nn.Linear(self.field_size*(self.field_size-1)/2, deep_layers[0])
        if self.is_batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
        if self.is_deep_dropout:
            self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

        print("Init deep part succeed")

        print "Init succeed"

    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        """
            fm part
        """
        if self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr,1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

            if self.interation_type:
                # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
                fm_second_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
                fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
                fm_sum_second_order_emb_square = fm_sum_second_order_emb*fm_sum_second_order_emb # (x+y)^2
                fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
                fm_second_order_emb_square_sum = sum(fm_second_order_emb_square) #x^2+y^2
                fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            else:
                fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                           enumerate(self.fm_second_order_embeddings)]
                fm_wij_arr = []
                for i in range(self.field_size):
                    for j in range(i + 1, self.field_size):
                        fm_wij_arr.append(fm_second_order_emb_arr[i] * fm_second_order_emb_arr[j])


        """
            ffm part
        """
        if self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr,1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:,i,:]), 1).t() * Xv[:,i]).t() for emb in  f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    ffm_wij_arr.append(ffm_second_order_emb_arr[i][j]*ffm_second_order_emb_arr[j][i])
            ffm_second_order = sum(ffm_wij_arr)

        """
            high interaction part
        """
        if self.use_high_interaction and self.use_fm:
            total_prod = 1.0
            for i, h_weight in enumerate(self.h_weights):
                total_prod = total_prod * (fm_second_order_emb_arr[i]*h_weight+self.h_bias[i])
            high_output = total_prod


        """
            deep part
        """
        if self.use_fm and self.interation_type:
            deep_emb = fm_second_order
        elif self.use_ffm and self.interation_type:
            deep_emb = ffm_second_order
        elif self.use_fm:
            deep_emb = torch.cat([torch.sum(fm_wij,1).view([-1,1]) for fm_wij in fm_wij_arr], 1)
        else:
            deep_emb = torch.cat([torch.sum(ffm_wij,1).view([-1,1]) for ffm_wij in ffm_wij_arr],1)

        if self.deep_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu

        if self.is_deep_dropout:
            deep_emb = self.linear_0_dropout(deep_emb)
        x_deep = self.linear_1(deep_emb)
        if self.is_batch_norm:
            x_deep = self.batch_norm_1(x_deep)
        x_deep = activation(x_deep)
        if self.is_deep_dropout:
            x_deep = self.linear_1_dropout(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)

        """
            sum
        """
        if self.use_fm:
            if self.use_high_interaction and not self.pre_train:
                total_sum = self.bias+ torch.sum(fm_first_order,1) + torch.sum(x_deep, 1) + torch.sum(high_output,1)
            else:
                total_sum = self.bias + torch.sum(fm_first_order, 1) + torch.sum(x_deep, 1)
        elif self.use_ffm:
            total_sum = self.bias + torch.sum(ffm_first_order, 1) + torch.sum(x_deep, 1)
        return total_sum
