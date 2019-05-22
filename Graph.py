# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:13:08 2019

@author: Hans
"""
import json
import numpy as np

seed = 131
mask=0x0fffffffffffffff

def BKDR2hash64v2(str):
    hash = 0
    for c in str:
        hash = hash*seed + ord(c)
    return hash&mask
    
class node():
    def __init__(self):
        self.items = []
        self.idx = 0
    def put(self, n1):
        if len(self.items)<10:
            self.items.append(n1)
            self.idx+=1
        else:
            self.items[self.idx]=n1
            self.idx+=1
        if self.idx==10:
            self.idx=0
    
class graph():
    def __init__(self):
        self.graph = {}
        self.st_features = {}
    def add_edge(self, n1, n2):
        if n1 not in self.graph:
            self.graph[n1] = node()
        self.graph[n1].put(n2)
        if n2 not in self.graph:
            self.graph[n2] = node()
        self.graph[n2].put(n1)
    def get_adj(self, n1):
        if n1 not in self.graph:
            self.graph[n1] = node()
        return self.graph[n1].items

class ps_server():
    def __init__(self, length):
        self.ps_data = {}
        self.length = length
    def get(self, key):
        if key not in self.ps_data:
            self.ps_data[key] = np.random.rand(self.length).astype(np.float32)*0.001
        return self.ps_data[key]
    def set_key(self, key, v):        
        self.ps_data[key] = v  

def create_graph_from_file(filename):
    g = graph()
    count=0
    with open(filename,"r", encoding='UTF-8') as fp:
        for line in fp:
            count+=1
            if count>10000:
                break
            td = json.loads(line)
            user = td['User']        
            uid=0
            user_st_feature = []
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
            g.st_features[uid]=user_st_feature
            
            doc = td['Doc']
            doc_st_feature = []
            did = 0
            for v in doc:
                hash_val = BKDR2hash64v2(v)
                if did == 0 and v.startswith("d_st_2_did"):
                    did = hash_val
                    doc_st_feature.append(hash_val)
                elif v.startswith("d_st"):
                    doc_st_feature.append(hash_val)
            g.st_features[did] = doc_st_feature
    return g
                
                
                