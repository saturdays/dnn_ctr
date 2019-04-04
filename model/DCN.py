# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of NFM

Reference:
[1] Deep & Cross Network for Ad Click Predictions
Ruoxi Wang,Stanford University,Stanford, CA,ruoxi@stanford.edu
Bin Fu,Google Inc.,New York, NY,binfu@google.com
Gang Fu,Google Inc.,New York, NY,thomasfu@google.com
Mingliang Wang,Google Inc.,New York, NY,mlwang@google.com

"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
"""
    网络结构部分
"""

class DCN(torch.nn.Module):
    def __init__(self,field_size, feature_sizes, embedding_size = 4,
                 h_depth = 2, deep_layers = [32, 32], is_deep_dropout = True, dropout_deep=[0.0, 0.5, 0.5],
                 h_cross_depth = 3,
                 h_inner_product_depth = 2, inner_product_layers = [32, 32], 
                 is_inner_product_dropout = True, dropout_inner_product_deep = [0.0, 0.5, 0.5],
                 deep_layers_activation = 'relu',is_batch_norm = False,                  
                 use_cross = True, use_inner_product = False, use_deep = True
                 ):
        super(DCN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.h_cross_depth = h_cross_depth
        self.h_inner_product_depth = h_inner_product_depth
        self.inner_product_layers = inner_product_layers
        self.is_inner_product_dropout = is_inner_product_dropout
        self.dropout_inner_product_deep = dropout_inner_product_deep
        self.deep_layers_activation = deep_layers_activation
        self.is_batch_norm = is_batch_norm
        self.use_cross = use_cross
        self.use_inner_product = use_inner_product
        self.use_deep = use_deep

        torch.manual_seed(self.random_seed)
        """
            check model type
        """
        if self.use_cross and self.use_deep and self.use_inner_product:
            print("The model is (cross network + deep network + inner_product network)")
        elif self.use_cross and self.use_deep:
            print("The model is (cross network + deep network)")
        elif self.use_cross and self.use_inner_product:
            print("The model is (cross network + inner product network)")
        elif self.use_inner_product and self.use_deep:
            print("The model is (inner product network + deep network)")
        elif self.use_cross:
            print("The model is a cross network only")
        elif self.use_deep:
            print("The model is a deep network only")
        elif self.use_inner_product:
            print("The model is an inner product network only")
        else:
            print("You have to choose more than one of (cross network, deep network, inner product network) models to use")
            exit(1)

        """
            embeddings
        """
        self.embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        cat_size = 0
        """
            cross part
        """
        if self.use_cross:
            print("Init cross network")
            for i in range(self.h_cross_depth):
                setattr(self, 'cross_weight_' + str(i+1),
                        torch.nn.Parameter(torch.randn(self.field_size*self.embedding_size)))
                setattr(self, 'cross_bias_' + str(i + 1),
                        torch.nn.Parameter(torch.randn(self.field_size * self.embedding_size)))
            print("Cross network finished")
            cat_size += self.field_size * self.embedding_size

        """
            inner prodcut part
        """
        if self.use_inner_product:
            print("Init inner product network")
            if self.is_inner_product_dropout:
                self.inner_product_0_dropout = nn.Dropout(self.dropout_inner_product_deep[0])
            self.inner_product_linear_1 = nn.Linear(self.field_size*(self.field_size-1)/2, self.inner_product_layers[0])
            if self.is_inner_product_dropout:
                self.inner_product_1_dropout = nn.Dropout(self.dropout_inner_product_deep[1])
            if self.is_batch_norm:
                self.inner_product_batch_norm_1 = nn.BatchNorm1d(self.inner_product_layers[0])

            for i, h in enumerate(self.inner_product_layers[1:], 1):
                setattr(self, 'inner_product_linear_' + str(i + 1), nn.Linear(self.inner_product_layers[i - 1], self.inner_product_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'inner_product_batch_norm_' + str(i + 1), nn.BatchNorm1d(self.inner_product_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'inner_product_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_inner_product_deep[i + 1]))
            cat_size += inner_product_layers[-1]
            print("Inner product network finished")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
            self.linear_1 = nn.Linear(self.embedding_size*self.field_size, deep_layers[0])
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
            cat_size += deep_layers[-1]
            print("Init deep part succeed")
        self.last_layer = nn.Linear(cat_size,1)

    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """

        if self.deep_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu

        """
            embeddings
        """
        emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.embeddings)]
        outputs = []
        """
            cross part
        """
        if self.use_cross:
            x_0 = torch.cat(emb_arr,1)
            x_l = x_0
            for i in range(self.h_cross_depth):
                x_l = torch.sum(x_0 * x_l, 1).view([-1,1]) * getattr(self,'cross_weight_'+str(i+1)).view([1,-1]) + getattr(self,'cross_bias_'+str(i+1)) + x_l
            outputs.append(x_l)

        """
            inner product part
        """
        if self.use_inner_product:
            fm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    fm_wij_arr.append(torch.sum(emb_arr[i] * emb_arr[j],1).view([-1,1]))
            inner_output = torch.cat(fm_wij_arr,1)

            if self.is_inner_product_dropout:
                deep_emb = self.inner_product_0_dropout(inner_output)
            x_deep = self.inner_product_linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.inner_product_batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_inner_product_dropout:
                x_deep = self.inner_product_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'inner_product_linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'inner_product_batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'inner_product_' + str(i + 1) + '_dropout')(x_deep)
            outputs.append(x_deep)

        """
            deep part
        """
        if self.use_deep:
            deep_emb = torch.cat(emb_arr,1)

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
            outputs.append(x_deep)

        """
            total
        """
        output = self.last_layer(torch.cat(outputs,1))
        return torch.sum(output,1)
