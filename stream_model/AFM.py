import torch
import torch.nn as nn
from torch.autograd import Function
from stream_model import FM_func

class AFM(torch.nn.Module):
    def __init__(self, vl=32):
        super(AFM, self).__init__()
    def forward(self, nodes, user, contex, doc):#input dictionary
        #create level0 features from base features
        #ft_lv0 = {n0:FM_func.apply(node_tensors[n0]) for n0 in node_tensors.keys()}
        pass