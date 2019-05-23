import torch
import torch.nn as nn
from torch.autograd import Function

class Sum_Cross(Function):
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

class FM_Cross_Func(Function):
    @staticmethod
    def forward(ctx, features):
        order1 = torch.sum(features, dim=0)
        order2 = torch.sum(features*features, dim=0)
        order2 = 0.5*(order1*order1-order2)
        ctx.save_for_backward(features, order1)
        return torch.cat([order2], dim=-1)
    @staticmethod
    def backward(ctx, grad_output):
        features, order1 = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            grad = grad_output*(order1 - features)
        return grad