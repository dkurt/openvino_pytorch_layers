import numpy as np
import torch
from torch import nn
from .lstsq import LSTSQ

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return LSTSQ.apply(B, A)


# Solves min_X||AX - B|| where A has a shape Mx2 and B has a shape MxN
def export(M, N):
    np.random.seed(324)
    torch.manual_seed(32)

    model = Model()
    A = torch.rand([M, 2])
    B = torch.rand([M, N])

    with torch.no_grad():
        torch.onnx.export(model, (A, B), 'model.onnx',
                          input_names=['input', 'input1'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(A, B)
    np.save('inp', A.detach().numpy())
    np.save('inp1', B.detach().numpy())
    np.save('ref', ref.detach().numpy())
