import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .deformable_conv import DeformableConvolution

np.random.seed(324)
torch.manual_seed(32)


class MyModel(nn.Module):
    def __init__(
        self,
        inplanes,
        outplanes,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        deformable_groups=1,
    ):
        super(MyModel, self).__init__()
        self.def_conv = DeformableConvolution(
            inplanes,
            outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=deformable_groups,
        )

    def forward(self, x, offset):
        y = self.def_conv(x, offset)
        return y


def export(
    inplanes=15, outplanes=15, inp_shape=[4, 3, 10, 10], offset_shape=[4, 18, 8, 8]
):
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel(inplanes, outplanes)
    model.eval()

    x = Variable(torch.randn(inp_shape))
    offset = Variable(torch.randn(offset_shape))
    ref = model(x, offset)

    np.save("inp", x.detach().numpy())
    np.save("inp1", offset.detach().numpy())
    np.save("ref", ref.detach().numpy())

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, offset),
            "model.onnx",
            input_names=["input", "input1"],
            output_names=["output"],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            opset_version=12,
        )

    # from openvino.inference_engine import IECore
    # ie = IECore()
    # net = ie.read_network('model.onnx')
    # exec_net = ie.load_network(net, 'CPU')

    # inputs = {'input':x, 'input1':offset}
    # out = exec_net.infer(inputs)
    # out = next(iter(out.values()))

    # diff = np.max(np.abs(ref.detach().numpy() - out))
    # print(diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ONNX model and test data")
    parser.add_argument("--inp_shape", type=int, nargs="+", default=[1, 15, 128, 240])
    parser.add_argument(
        "--offset_shape", type=int, nargs="+", default=[1, 18, 128, 240]
    )
    parser.add_argument("--inplanes", type=int, nargs="+", default=15)
    parser.add_argument("--outplanes", type=int, nargs="+", default=15)
    args = parser.parse_args()

    export(args.inplanes, args.outplanes, args.inp_shape, args.offset_shape)
