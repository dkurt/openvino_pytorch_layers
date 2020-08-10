import torch
import torch.nn as nn

class Unpool2d(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, indices):
        return g.op('Unpooling', x, indices)

    @staticmethod
    def forward(self, x, indices):
        return nn.MaxUnpool2d(2, stride=2)(x, indices)
