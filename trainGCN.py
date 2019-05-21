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

filename = 'C:/Users/Hans/init_graph.list'
g = create_graph_from_file(filename)
vector_length = 32
ps_client = ps_server(vector_length*2)#value and grad

filename = 'C:/Users/Hans/train_data.list'

class FM_func(Function):
    @staticmethod
    def forward(ctx, all_tensors):
        order1 = torch.sum(all_tensors, dim=0)
        order2 = torch.sum(all_tensors*all_tensors, dim=0)
        order2 = 0.5*(order1*order1-order2)
        ctx.save_for_backward(all_tensors, order1)
        return torch.cat([order1, order2], dim=-1)
    @staticmethod
    def backward(ctx, grad_output):
        all_tensors, order1 = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            output_len = grad_output.shape[-1]//2
            grad1 = grad_output[...,0:output_len]
            gard2 = grad_output[..., output_len:]
            grad = grad1 + gard2*(order1 - all_tensors)
        return grad

class GCN(torch.nn.Module):
    def __init__(self, graph, vl=32):
        super(GCN, self).__init__()
        self.vl = vl
        self.fc1 = nn.Linear(4*self.vl, 2*self.vl, bias=False)
        self.fc2 = nn.Linear(4*self.vl, 2*self.vl, bias=False)
        self.graph = graph
        self.fc1.weight.data.fill_(0.01) 
        self.fc2.weight.data.fill_(0.01) 
    def conv1(self, n1, ft_lv0):
        node_0 = ft_lv0[n1]
        if g.get_adj(n1):
            pool_adj = torch.stack([ft_lv0[n0] for n0 in g.get_adj(n1)])
            pool_adj = torch.sum(pool_adj, dim=0) 
            cross = node_0*pool_adj
            z = torch.cat((node_0+pool_adj, cross), dim=-1) 
        else:
            pool_adj = torch.zeros(2*self.vl)
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
            pool_adj = torch.zeros(2*self.vl)            
            z = torch.cat((node_1, pool_adj), dim=-1)     
        return z
    def forward(self, node_l2, node_l1, ft_lv0):
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
batch = 2

#create model Input   
#def forward(self, node, adj, adj_d1):  
#需要使用python访问ps。或者rpc传输参数
#这个交叉项求导消耗了9秒
def gen_node_input2(node, aux_ft, all_tensors):
    order_1 = torch.zeros(vector_length)
    order_2 = torch.zeros(vector_length)
    for key in g.st_features.keys():
        if key not in all_tensors:        
            ft = ps_client.get(key)
            all_tensors[key] = torch.tensor(ft[0:vector_length]).requires_grad_()
        order_1 += all_tensors[key]
        order_2 += all_tensors[key]*all_tensors[key]
    for key in aux_ft:
        if key not in all_tensors:
            ft = ps_client.get(key)
            all_tensors[key] = torch.tensor(ft[0:vector_length]).requires_grad_()
        order_1 += all_tensors[key]
        order_2 += all_tensors[key]*all_tensors[key]
    return torch.cat([order_1, 0.5*(order_1*order_1-order_2)])

def gen_node_input(node, aux_ft, all_tensors):
    features = []
    for key in g.st_features.keys():
        if key not in all_tensors:        
            ft = ps_client.get(key)
            all_tensors[key] = torch.tensor(ft[0:vector_length]).requires_grad_()
        features.append(all_tensors[key])
    for key in aux_ft:
        if key not in all_tensors:
            ft = ps_client.get(key)
            all_tensors[key] = torch.tensor(ft[0:vector_length]).requires_grad_()
        features.append(all_tensors[key])
    ft = torch.stack(features)
    return FM_func.apply(ft)

def train(uid, user_dy, did, doc_dy, target):
    node_l2 = [uid, did]
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
    #create Tensors
    all_tensors ={}    
    ft_lv0 = {}
    for v in node_l0:
        if v == uid:
            ft_lv0[v] = gen_node_input(v, user_dy, all_tensors)
        elif v==did:
            ft_lv0[v] = gen_node_input(v, doc_dy, all_tensors)
        else:            
            ft_lv0[v] = gen_node_input(v, [], all_tensors)   
    ft_lv2 = model(node_l2, node_l1, ft_lv0)
    
    node_l2_idx = {n1:idx for idx, n1 in enumerate(node_l2)}
    output = torch.sum(ft_lv2[node_l2_idx[did]]*ft_lv2[node_l2_idx[uid]])
    output = output.expand(1)
    loss = criterion(output, target)
    loss.backward()

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
    start = time.time()
    all_tensors ={}
    ft_lv0 = {}
    for idx, v in enumerate(batch_uid):
        if v not in ft_lv0:
            ft_lv0[v] = gen_node_input(v, batch_user_dy[idx], all_tensors)    
    for idx, v in enumerate(batch_did):
        if v not in ft_lv0:
            ft_lv0[v] = gen_node_input(v, batch_doc_dy[idx], all_tensors)        
    for v in node_l0:
        if v not in ft_lv0:
            ft_lv0[v] = gen_node_input(v, [], all_tensors)            
    ft_lv2 = model(node_l2, node_l1, ft_lv0)
    print('forward time:', time.time()-start)
    output = torch.sum(ft_lv2[0:batch]*ft_lv2[batch:], dim=-1)
    loss = criterion(output, target)
    start = time.time()
    loss.backward()
    print('backward time:', time.time()-start)
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
                break
