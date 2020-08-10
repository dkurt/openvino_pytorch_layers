import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from unpool import Unpool2d

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.unpool = Unpool2d()

    def forward(self, x):
        output, indices = self.pool(x)
        conv = self.conv(output)
        return self.unpool.apply(conv, indices)


inp = Variable(torch.randn(5, 3, 10, 12))
model = MyModel()
model.eval()

with torch.no_grad():
    torch.onnx.export(model, inp, 'model_with_unpool.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

ref = model(inp)
np.save('inp', inp.detach().numpy())
np.save('ref', ref.detach().numpy())
