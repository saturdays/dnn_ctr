import torch
import torch.nn as nn
from torch.autograd import Function
from stream_model import Lin_Cross_func

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
        ft_lv0 = {n0:Sum_Cross.apply(node_tensors[n0]) for n0 in node_tensors.keys()}
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