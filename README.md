Repository with guides to enable some layers from PyTorch in Intel OpenVINO:

[![CI](https://github.com/dkurt/openvino_pytorch_layers/workflows/CI/badge.svg?branch=master)](https://github.com/dkurt/openvino_pytorch_layers/actions?query=branch%3Amaster)

* [nn.MaxUnpool2d](examples/unpool)
* [torch.fft](examples/fft)
* [nn.functional.grid_sample](https://github.com/dkurt/openvino_pytorch_layers/tree/master/examples/grid_sample)
* [torchvision.ops.DeformConv2d](examples/deformable_conv)
* [SparseConv](examples/sparse_conv) from [Open3D](https://github.com/isl-org/Open3D)
* [torch.lstsq](https://pytorch.org/docs/stable/generated/torch.lstsq.html)


## OpenVINO Model Optimizer extension

To create OpenVINO IR, use extra `--extension` flag to specify a path to Model Optimizer extensions that perform graph transformations and register custom layers.

```bash
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model model.onnx \
    --extension openvino_pytorch_layers/mo_extensions
```

## Custom CPU extensions

You also need to build CPU extensions library which actually has C++ layers implementations:
```bash
source /opt/intel/openvino/bin/setupvars.sh
export TBB_DIR=/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/cmake/

cd user_ie_extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc --all)
```

Add compiled extensions library to your project:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.add_extension('user_ie_extensions/build/libuser_cpu_extension.so', 'CPU')

net = ie.read_network('model.xml', 'model.bin')
exec_net = ie.load_network(net, 'CPU')
```
