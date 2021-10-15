import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from fft import FFT

np.random.seed(324)
torch.manual_seed(32)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fft = FFT()

    def forward(self, x):
        y = self.fft.apply(x, False)
        y = y * 2
        # TODO: there is a bug with "inverse" data attribute in OpenVINO 2021.4
        # y = self.fft.apply(y, True)
        return y

parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
parser.add_argument('--shape', type=int, nargs='+', default=[5, 3, 6, 8, 2])
args = parser.parse_args()

model = MyModel()
inp = Variable(torch.randn(args.shape))
model.eval()

with torch.no_grad():
    torch.onnx.export(model, inp, 'model.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

ref = model(inp)
np.save('inp', inp.detach().numpy())
np.save('ref', ref.detach().numpy())
