import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from stream_model.FM_func import FM_Cross_Func

class FFM_model(torch.nn.Module):
    def __init__(self, vl=32):
        super(FFM_model, self).__init__()
        self.vl = vl
    def forward(self, user_list, ctx_list, doc_list):#features list for each training sample
        output = []
        for idx in range(len(user_list)):            
            user = user_list[idx]
            ctx = ctx_list[idx]
            doc = doc_list[idx]
            z = FM_Cross_Func.apply(user[:,0:self.vl]).sum()
            z += FM_Cross_Func.apply(ctx[:,self.vl:2*self.vl]).sum()
            z += FM_Cross_Func.apply(doc[:,2*self.vl:3*self.vl]).sum()
            z += torch.dot( (user[:,self.vl:2*self.vl]).sum(0), (ctx[:,0:self.vl]).sum(0) )
            z += torch.dot( (user[:,2*self.vl:3*self.vl]).sum(0), (doc[:,0:self.vl]).sum(0) )
            z += torch.dot( (ctx[:,2*self.vl:3*self.vl]).sum(0), (doc[:,self.vl:2*self.vl]).sum(0) )
            z += user[:,3*self.vl].sum() + ctx[:,3*self.vl].sum() + doc[:,3*self.vl].sum()
            y = torch.sigmoid(z)
            output.append(y)
        return torch.stack(output)
    