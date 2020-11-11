import torch

class FFT(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, inverse):
        return g.op('FFT', x,
                    inverse_i=inverse)

    @staticmethod
    def forward(self, x, inverse):
        # https://pytorch.org/docs/stable/torch.html#torch.fft
        if inverse:
            y = torch.ifft(input=x, signal_ndim=2, normalized=True)
        else:
            y = torch.fft(input=x, signal_ndim=2, normalized=True)
        return y
