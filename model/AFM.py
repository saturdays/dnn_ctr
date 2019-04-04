# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of AFM

Reference:
[1] Attentional Factorization Machines:Learning theWeight of Feature Interactions via Attention Networks

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
"""
    网络结构部分
"""

class AFM(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    attention_size: The attention netwotk's parameter
    dropout_shallow: an array of the size of 1, example:[0.5], the element is for the-first order part
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    is_batch_norm：bool,  use batch_norm or not
    Attention: only support logsitcs regression
    """
    def __init__(self,field_size, feature_sizes, embedding_size = 4, attention_size = 4, 
                 dropout_shallow = -1.0,
                 dropout_attention = -1.0,
                 attention_layers_activation = 'relu', 
                 compression = 0,
                 is_batch_norm = False,
                 use_fm = True, use_ffm = False           
                 ):
        super(AFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.attention_size = attention_size

        self.dropout_shallow = dropout_shallow
        self.dropout_attention = dropout_attention
        self.attention_layers_activation = attention_layers_activation
        self.compression = compression
        self.is_batch_norm = is_batch_norm
        self.use_fm = use_fm
        self.use_ffm = use_ffm

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is afm(fm+attention layers)")
        elif self.use_ffm:
            print("The model is affm(ffm+attention layers)")
        else:
            print("You have to choose more than one of (fm, ffm) models to use")
            exit(1)
        """
            bias
        """
        self.bias = torch.nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(0, 0.2)
        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow>0.0:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow)
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            print("Init fm part succeed")
        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            if self.dropout_shallow>0.0:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow)
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            print("Init ffm part succeed")
        """
            attention part
        """
        print("Init attention part")

        if self.dropout_attention>0.0:
            self.attention_linear_0_dropout = nn.Dropout(self.dropout_attention)
        self.attention_linear_1 = nn.Linear(self.embedding_size, self.attention_size)
        self.H = torch.nn.Parameter(torch.Tensor(self.attention_size))
        self.P = torch.nn.Parameter(torch.Tensor(self.embedding_size))
        self.H.data.normal_(0, 0.2)
        self.P.data.normal_(0, 0.2)

        if self.compression > 0:
            self.conv1 = torch.nn.Conv1d(self.field_size, self.compression , kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            self.bn1 = torch.nn.BatchNorm1d(self.compression, momentum=None)
            print("Init attention part succeed")
            print("Init succeed")
        if self.is_batch_norm:
            self.shallow_bn = torch.nn.BatchNorm1d(self.field_size, momentum=None)

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
            if self.dropout_shallow>0.1:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)
            elif self.is_batch_norm:
                fm_first_order = self.shallow_bn(fm_first_order)

            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                        enumerate(self.fm_second_order_embeddings)]
            if self.compression > 0:
                fm_second_order_emb_arr = torch.stack(fm_second_order_emb_arr, dim=1)                
                fm_second_order_emb_arr = self.conv1(fm_second_order_emb_arr)
                fm_second_order_emb_arr = self.bn1(fm_second_order_emb_arr)
                fm_second_order_emb_arr = F.relu(fm_second_order_emb_arr)
                self.field_size = self.compression
                fm_second_order_emb_arr = fm_second_order_emb_arr.permute(1,0,2)
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
            if self.dropout_shallow>0.0:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:,i,:]), 1).t() * Xv[:,i]).t() for emb in  f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    ffm_wij_arr.append(ffm_second_order_emb_arr[i][j]*ffm_second_order_emb_arr[j][i])

        """
            attention part
        """
        if self.use_fm:
            interaction_layer = torch.cat(fm_wij_arr, 1)
        else:
            interaction_layer = torch.cat(ffm_wij_arr,1)
        """
        if self.attention_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.attention_layers_activation == 'tanh':
            activation = F.tanh
        elif self.attention_layers_activation == 'leakyrelu':
            activation = F.leaky_relu
        else:
            activation = F.relu
        """
        if self.dropout_attention>0.0:
            interaction_layer = self.attention_linear_0_dropout(interaction_layer)
        attention_tmp = self.attention_linear_1(interaction_layer.view([-1,self.embedding_size]))
        attention_tmp = attention_tmp * self.H
        attention_tmp = torch.sum(attention_tmp,1).view([-1,self.field_size*(self.field_size-1)//2])
        attention_weight = torch.nn.Softmax(dim=1)(attention_tmp)
        attention_output = torch.sum(interaction_layer.view([-1,self.embedding_size])*self.P,1).view([-1,self.field_size*(self.field_size-1)//2])
        attention_output = attention_output * attention_weight
        """
            sum
        """
        if self.use_fm:
            total_sum = self.bias+ torch.sum(fm_first_order,1) + torch.sum(attention_output,1)
        elif self.use_ffm:
            total_sum = self.bias + torch.sum(ffm_first_order, 1) + torch.sum(attention_output, 1)
        return total_sum
