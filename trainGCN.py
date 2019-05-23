import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json
from stream_model.Graph import graph, create_graph_from_file
from stream_model.param_server import ps_server, BKDR2hash64v2
from stream_model import GCN

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
model = GCN(g, vector_length)
Parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adagrad(Parameters, 0.01, weight_decay=1e-5)
criterion = nn.BCELoss()
batch = 4
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
        nan_count = np.sum(np.isnan(node_grad))
        if nan_count>0:
            print('nan grad ', nan_count)
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
    loss = criterion(output, target)/batch
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
