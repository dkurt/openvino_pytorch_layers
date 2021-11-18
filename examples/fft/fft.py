import torch

class FFT(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, inverse):
        return g.op('IFFT' if inverse else 'FFT', x,
                    inverse_i=inverse)

    @staticmethod
    def forward(self, x, inverse):
        # https://pytorch.org/docs/stable/torch.html#torch.fft
        signal_ndim = 2 if len(x.shape) == 5 else 1
        if inverse:
            y = torch.ifft(input=x, signal_ndim=signal_ndim, normalized=True)
        else:
            y = torch.fft(input=x, signal_ndim=signal_ndim, normalized=True)
        return y
