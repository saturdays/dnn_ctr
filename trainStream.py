import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from stream_model.FM import FM_model
from stream_model.FFM import FFM_model
from stream_model.param_server import ps_server, BKDR2hash64v2

import torch
import torch.nn as nn
import torch.optim

vector_length = 32
method = 'FFM'
if method == 'FM':
    weight_len = vector_length+1
    ps_client = ps_server(weight_len*2)#value and grad
    model = FM_model(vector_length)
elif method == 'FFM':
    weight_len = 3*vector_length+1
    ps_client = ps_server(weight_len*2)#value and grad
    model = FFM_model(vector_length)

filename = 'C:/Users/Hans/train_data.list'
#Parameters = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = torch.optim.Adagrad(Parameters, 0.01, weight_decay=1e-5)
criterion = nn.BCELoss()
batch = 200
MAXGDSUM=100000000
alpha=0.999
lr=0.01
lambda2=1
#L2 = 0.0

#create model Input
def get_key_features(all_keys, key_features):
    for keys in all_keys:
        for key in keys:
            if key in key_features:
                continue
            key_features[key]=ps_client.get(key)

def update_key_features(key_features, key_grads):
    for key, grad in key_grads.items():
        grad[grad>MAXGDSUM] = MAXGDSUM
        key_features[key][weight_len:]=alpha*key_features[key][weight_len:]+grad*grad
        tmp = np.sqrt(key_features[key][weight_len:])
        key_features[key][0:weight_len] -= lr*(grad/(lambda2+tmp))
    for key, ft in key_features.items():
        ps_client.set_key(key, ft)

def create_tensors(batch_keys, key_features):
    node_tensors=[]
    for keys in batch_keys:        
        features = np.zeros((len(keys), weight_len)).astype(np.float32)
        for idx, key in enumerate(keys):
            features[idx] = key_features[key][0:weight_len]
        node_tensors.append(torch.tensor(features).requires_grad_())
    return node_tensors

def calculate_key_grad(batch_keys, batch_tensors, key_grads):
    for idx, keys in enumerate(batch_keys):
        ft_grad = batch_tensors[idx].grad.numpy()
        ft_grad = np.nan_to_num(ft_grad)
        for jdx, key in enumerate(keys):
            if key not in key_grads:
                key_grads[key] = np.zeros(weight_len).astype(np.float32)
            key_grads[key] += ft_grad[jdx]
    return key_grads
    
def batch_train(batch_user_features, batch_contex_features, batch_doc_features, target):
    key_features = {}
    get_key_features(batch_user_features, key_features)
    get_key_features(batch_contex_features, key_features)
    get_key_features(batch_doc_features, key_features)

    user_tensors = create_tensors(batch_user_features, key_features)
    ctx_tensors = create_tensors(batch_contex_features, key_features)
    doc_tensors = create_tensors(batch_doc_features, key_features)
    #print('create time:', time.time()-start)  
    output = model(user_tensors, ctx_tensors, doc_tensors)
    loss = criterion(output, target)
    print('loss ', loss.item())
    loss.backward()
    key_grads = {}
    calculate_key_grad(batch_user_features, user_tensors, key_grads)
    calculate_key_grad(batch_contex_features, ctx_tensors, key_grads)
    calculate_key_grad(batch_doc_features, doc_tensors, key_grads)
    update_key_features(key_features, key_grads)
    pass

with open(filename,"r", encoding='UTF-8') as fp:
        batch_user_features = []
        batch_contex_features = []
        batch_doc_features = []
        batch_target = []
        count = 0
        for line in fp:
            user_features = []
            contex_features = []
            doc_features = []
            td = json.loads(line)
            for v in td['User']:
                hash_val = BKDR2hash64v2(v)
                user_features.append(hash_val)
            for v in td['Context']:
                hash_val = BKDR2hash64v2(v)
                contex_features.append(hash_val)
            for v in td['Doc']:
                hash_val = BKDR2hash64v2(v)
                doc_features.append(hash_val)
            
            batch_user_features.append(user_features)
            batch_contex_features.append(contex_features)
            batch_doc_features.append(doc_features)
            batch_target.append(td['Y'])
            count+=1

            if count==batch:
                target= torch.tensor(batch_target, dtype=torch.float)
                batch_train(batch_user_features, batch_contex_features, batch_doc_features, target)
                batch_user_features = []
                batch_contex_features = []
                batch_doc_features = []
                batch_target = []
                count=0
