import torch
import torch.nn as nn

class Unpool2d(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, indices, output_size=None):
        return g.op('Unpooling', x, indices,
                    output_size_i=output_size)

    @staticmethod
    def forward(self, x, indices, output_size=None):
        return nn.MaxUnpool2d(2, stride=2)(x, indices, output_size=output_size)
