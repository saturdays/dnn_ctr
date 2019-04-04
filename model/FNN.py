# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of FNN

Reference:
[1] Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction

Weinan Zhang, Tianming Du, Jun Wang

"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
"""
    网络结构部分
"""

class FNN(torch.nn.Module):

    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth=2, deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='tanh', 
                 optimizer_type='adam', is_batch_norm=False, 
                 use_fm=True, use_ffm=False
                 ):
        super(FNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.pretrain = False

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is FNN(fm+nn layers)")
        elif self.use_ffm:
            print("The model is FFNN(ffm+nn layers)")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True) #w0
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes]) #wi
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]) #vi
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            print("Init ffm part succeed")

        print("Init nn part")
        if self.is_deep_dropout:
            self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
        if not use_ffm:
            self.linear_1 = nn.Linear(1 + self.field_size + self.field_size  * self.embedding_size, deep_layers[0])
        else:
            self.linear_1 = nn.Linear(1 + self.field_size + self.field_size * self.field_size * self.embedding_size, deep_layers[0])

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
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)
        print("Init nn part succeed")

    def forward(self, Xi, Xv):
        """
        :param Xi: index input tensor, batch_size * k * 1
        :param Xv: value input tensor, batch_size * k * 1
        :param is_pretrain: the para to decide fm pretrain or not
        :return: the last output
        """
        if self.pretrain and self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_first_order_sum = torch.sum(sum(fm_first_order_emb_arr),1)
            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
            fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            fm_sum_second_order_emb_square = fm_sum_second_order_emb*fm_sum_second_order_emb # (x+y)^2
            fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
            fm_second_order_emb_square_sum = sum(fm_second_order_emb_square) #x^2+y^2
            fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            fm_second_order_sum = torch.sum(fm_second_order,1)
            return self.fm_bias+fm_first_order_sum+fm_second_order_sum
        elif self.pretrain and self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            sum_ = torch.sum(sum(ffm_first_order_emb_arr),1)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:,i,:]), 1).t() * Xv[:,i]).t() for emb in  f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    sum_ += torch.sum((ffm_second_order_emb_arr[i][j]*ffm_second_order_emb_arr[j][i]),1)
            return self.ffm_bias + sum_
        elif not self.pretrain and self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr,1)
            fm_second_order = torch.cat(fm_second_order_emb_arr,1)
            if self.use_cuda:
                fm_bias = self.fm_bias * Variable(torch.ones(Xi.data.shape[0],1)).cuda()
            else:
                fm_bias = self.fm_bias * Variable(torch.ones(Xi.data.shape[0], 1))
            deep_emb = torch.cat([fm_bias,fm_first_order,fm_second_order],1)
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
            x_deep = self.deep_last_layer(x_deep)
            return torch.sum(x_deep,1)
        else:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            ffm_second_order_emb_arr = [torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs],1) for
                                        i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr,1)
            ffm_second_order = torch.cat(ffm_second_order_emb_arr,1)
            if self.use_cuda:
                ffm_bias = self.ffm_bias * Variable(torch.ones(Xi.data.shape[0], 1)).cuda()
            else:
                ffm_bias = self.ffm_bias * Variable(torch.ones(Xi.data.shape[0], 1))
            deep_emb = torch.cat([ffm_bias, ffm_first_order, ffm_second_order], 1)
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
            x_deep = self.deep_last_layer(x_deep)
            return torch.sum(x_deep,1)