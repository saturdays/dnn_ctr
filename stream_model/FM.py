import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from stream_model.FM_func import FM_Cross_Func

class FM_model(torch.nn.Module):
    def __init__(self, vl=32):
        super(FM_model, self).__init__()
        self.vl = vl
    def forward(self, user, ctx, doc):#features list for each training sample
        output = []
        for idx in range(len(user)):
            features = torch.cat([user[idx], ctx[idx], doc[idx]], dim=0)
            y = F.sigmoid(torch.sum(FM_Cross_Func.apply(features[:,0:self.vl]))
                        + torch.sum(features[:,-1]))
            output.append(y)
        return torch.stack(output)
    