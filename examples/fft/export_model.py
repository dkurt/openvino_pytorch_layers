import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .fft import FFT


class MyModel(nn.Module):
    def __init__(self, inverse):
        super(MyModel, self).__init__()
        self.fft = FFT()
        self.inverse = inverse

    def forward(self, x):
        return self.fft.apply(x, self.inverse)

def export(shape, inverse):
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel(inverse)
    inp = Variable(torch.randn(shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, inp, 'model.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp)
    np.save('inp', inp.detach().numpy())
    np.save('ref', ref.detach().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--shape', type=int, nargs='+', default=[5, 3, 6, 8, 2])
    args = parser.parse_args()

    export(args.shape)
