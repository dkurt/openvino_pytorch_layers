import torch

class FFT(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op('FFT', x)

    @staticmethod
    def forward(self, x):
        # https://pytorch.org/docs/stable/torch.html#torch.fft
        y = torch.fft(input=x, signal_ndim=2, normalized=True)
        return y
