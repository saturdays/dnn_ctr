import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json
from Graph import graph, create_graph_from_file, ps_server, BKDR2hash64v2

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch.optim
import time
from torch.autograd import Function

#filename = 'C:/Users/Hans/init_graph.list'
#g = create_graph_from_file(filename)
#vector_length = 32
#ps_client = ps_server(vector_length*2)#value and grad

filename = 'C:/Users/Hans/train_data.list'

class FM_func(Function):
    @staticmethod
    def forward(ctx, features):
        order1 = torch.sum(features, dim=0)
        order2 = torch.sum(features*features, dim=0)
        order2 = 0.5*(order1*order1-order2)
        ctx.save_for_backward(features, order1)
        return torch.cat([order1, order2], dim=-1)
    @staticmethod
    def backward(ctx, grad_output):
        features, order1 = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            output_len = grad_output.shape[-1]//2
            grad1 = grad_output[...,0:output_len]
            gard2 = grad_output[..., output_len:]
            grad = grad1 + gard2*(order1 - features)
        return grad

class GCN(torch.nn.Module):
    def __init__(self, gh, vl=32):
        super(GCN, self).__init__()
        self.vl = vl
        self.fc1 = nn.Linear(4*self.vl, 2*self.vl)
        self.fc2 = nn.Linear(4*self.vl, 2*self.vl)
        self.g = gh
    def conv1(self, n1, ft_lv0):
        node_0 = ft_lv0[n1]
        if g.get_adj(n1):
            pool_adj = torch.stack([ft_lv0[n0] for n0 in g.get_adj(n1)])
            pool_adj = torch.sum(pool_adj, dim=0) 
            cross = node_0*pool_adj
            z = torch.cat((node_0+pool_adj, cross), dim=-1) 
        else:
            pool_adj = torch.zeros(node_0.shape)
            z = torch.cat((node_0, pool_adj), dim=-1) 
        return z
    def conv2(self, n2, ft_lv1, node_l1_idx):
        node_1 = ft_lv1[node_l1_idx[n2]]
        if g.get_adj(n2):
            pool_adj = torch.stack([ft_lv1[node_l1_idx[n1]] for n1 in g.get_adj(n2)])
            pool_adj = torch.sum(pool_adj, dim=0)            
            cross = node_1*pool_adj
            z = torch.cat((node_1+pool_adj, cross), dim=-1)  
        else:
            pool_adj = torch.zeros(node_1.shape)            
            z = torch.cat((node_1, pool_adj), dim=-1)     
        return z
    def forward(self, node_l2, node_l1, node_tensors):
        #create level0 features from base features
        ft_lv0 = {n0:FM_func.apply(node_tensors[n0]) for n0 in node_tensors.keys()}
        #node_l2 nodes that need claculate level2 feature
        node_l1_idx = {n1:idx for idx, n1 in enumerate(node_l1)}
        #calculate level1 features
        z = torch.stack([self.conv1(n1, ft_lv0) for n1 in node_l1])
        z = self.fc1(z)
        ft_lv1 = F.relu(z)
        #calculate level2 features
        z = torch.stack([self.conv2(n2, ft_lv1, node_l1_idx) for n2 in node_l2])
        z = self.fc2(z)
        z = F.relu(z)
        ft_lv2 = F.normalize(z, dim=-1)
        return ft_lv2

model = GCN(g, vector_length)
Parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adagrad(Parameters, 0.01, weight_decay=1e-5)
criterion = nn.BCELoss()
batch = 16
MAXGDSUM=100000000
alpha=0.999
lr=0.01
lambda2=1
#L2 = 0.0

#create model Input   
#def forward(self, node, adj, adj_d1):  
#需要使用python访问ps。或者rpc交换ps
def get_node_keys(node, aux_ft, node_keys):
    node_keys[node] = list(g.st_features.keys())+aux_ft

def get_key_features(node_keys):
    key_features = {}
    for _, keys in node_keys.items():
        for key in keys:
            if key in key_features:
                continue
            key_features[key]=ps_client.get(key)
    return key_features

def update_key_features(key_features, key_grads):
    for key, grad in key_grads.items():
        grad[grad>MAXGDSUM] = MAXGDSUM
        key_features[key][vector_length:]=alpha*key_features[key][vector_length:]+grad*grad
        tmp = np.sqrt(key_features[key][vector_length:])
        key_features[key][0:vector_length] -= lr*(grad/(lambda2+tmp))
    for key, ft in key_features.items():
        ps_client.set_key(key, ft)

def create_node_tensors(node_keys, key_features):
    node_tensors={}
    for node, keys in node_keys.items():
        features = np.zeros((len(keys), vector_length)).astype(np.float32)
        for idx, key in enumerate(keys):
            features[idx] = key_features[key][0:vector_length]
        node_tensors[node] = torch.tensor(features).requires_grad_()
    return node_tensors

def calculate_key_grad(node_keys, node_tensors):
    key_grads = {}
    for node, keys in node_keys.items():
        node_grad = node_tensors[node].grad.numpy()
        node_grad = np.nan_to_num(node_grad)
        for idx, key in enumerate(keys):
            if key not in key_grads:
                key_grads[key] = np.zeros(vector_length).astype(np.float32)
            key_grads[key] += node_grad[idx]
    return key_grads
    
#假设在一个batch内同一user的动态特征不会变化
def batch_train(batch_uid, batch_user_dy, batch_did, batch_doc_dy, target):
    node_l2 = batch_uid+batch_did
    node_l1 = set()
    node_l0 = set()
    for v in node_l2:
        node_l1.add(v)
        for adj in g.get_adj(v):
            node_l1.add(adj)
    for v in node_l1:
        node_l0.add(v)
        for adj in g.get_adj(v):
            node_l0.add(adj)
    node_l1 = list(node_l1)
    node_l0 = list(node_l0)
    
    node_keys = {}
    for idx, v in enumerate(batch_uid):
        if v not in node_keys:
            get_node_keys(v, batch_user_dy[idx], node_keys)    
    for idx, v in enumerate(batch_did):
        if v not in node_keys:
            get_node_keys(v, batch_doc_dy[idx], node_keys)        
    for v in node_l0:
        if v not in node_keys:
            get_node_keys(v, [], node_keys)
    
    key_features = get_key_features(node_keys)
    #start = time.time()
    node_tensors = create_node_tensors(node_keys, key_features)
    #print('create time:', time.time()-start)  
    ft_lv2 = model(node_l2, node_l1, node_tensors)
    output = torch.sum(ft_lv2[0:batch]*ft_lv2[batch:], dim=-1)
    loss = criterion(output, target)
    print('loss ', loss.item())
    loss.backward()
    key_grads = calculate_key_grad(node_keys, node_tensors)
    update_key_features(key_features, key_grads)
    optimizer.step()
    pass

with open(filename,"r", encoding='UTF-8') as fp:
        batch_uid = []
        batch_did = []
        batch_user_dy = []
        batch_doc_dy = []
        batch_target = []
        count = 0
        for line in fp:
            td = json.loads(line)
            user = td['User']        
            uid=0
            user_st_feature = []
            user_dy_feature = []
            for v in user:
                hash_val = BKDR2hash64v2(v)
                if uid==0 and v.startswith("u_st_2_uid"):
                    uid = hash_val
                    user_st_feature.append(uid)
                elif v.startswith("d_st_2_did"):
                    g.add_edge(uid, hash_val)
                elif v.startswith("u_dy_6_readlist3"):
                    g.add_edge(uid, hash_val)
                elif v.startswith("u_st"):
                    user_st_feature.append(hash_val)
                elif v.startswith("u_dy"):
                    user_dy_feature.append(hash_val)
            g.st_features[uid]=user_st_feature

            doc = td['Doc']
            doc_st_feature = []
            doc_dy_feature = []
            did = 0
            for v in doc:
                hash_val = BKDR2hash64v2(v)
                if did == 0 and v.startswith("d_st_2_did"):
                    did = hash_val
                    doc_st_feature.append(hash_val)
                elif v.startswith("d_st"):
                    doc_st_feature.append(hash_val)
                elif v.startswith("d_dy"):
                    doc_dy_feature.append(hash_val)
            g.st_features[did]=doc_st_feature
            
            batch_uid.append(uid)
            batch_did.append(did)
            batch_user_dy.append(user_dy_feature)
            batch_doc_dy.append(doc_dy_feature)
            batch_target.append(td['Y'])
            count+=1
            
            if count==batch:
                target= torch.tensor(batch_target, dtype=torch.float)
                batch_train(batch_uid, batch_user_dy, batch_did, batch_doc_dy, target)
                batch_uid = []
                batch_did = []
                batch_user_dy = []
                batch_doc_dy = []
                batch_target = []
                count=0                
