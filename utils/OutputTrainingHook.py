import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

class OutputTrainingHook(nn.Module):
    #This training hook captures and handles the gradients at the output of the network
    def __init__(self):
        super(OutputTrainingHook, self).__init__()


    def forward(self, input, grad_at_output):
        out = OutputHookFunction.apply(input, grad_at_output)
        return out

class OutputHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,  grad_at_output):
        ctx.grad_out = grad_at_output

        return input.clone()


    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output = ctx.grad_out
        grad_at_output[:grad_output.shape[0], :].data.copy_(grad_output.data)
        return grad_output, None


class GetOutputTrainingHook(nn.Module):
    def __init__(self):
        super(GetOutputTrainingHook, self).__init__()


    def forward(self, input, grad_at_output):
        out = GetOutputHookFunction.apply(input, grad_at_output)
        return out

class GetOutputHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_at_output):
        ctx.grad_out = grad_at_output

        return input.clone()


    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output = ctx.grad_out
        return grad_at_output, None