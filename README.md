Guide of how to enable PyTorch `nn.MaxUnpool2d` in Intel OpenVINO.

[![CI](https://github.com/dkurt/openvino_pytorch_unpool/workflows/CI/badge.svg?branch=master)](https://github.com/dkurt/openvino_pytorch_unpool/actions?query=branch%3Amaster)

## Description
There are two problems with OpenVINO and MaxUnpool at the moment of this guide creation:

* OpenVINO does not have Unpooling kernels
* PyTorch -> ONNX conversion is unimplemented for `nn.MaxUnpool2d`

So following this guide you will learn
* How to perform PyTorch -> ONNX conversion for unsupported layers
* How to convert ONNX to OpenVINO Intermediate Respresentation (IR) with extensions
* How to write custom CPU layers in OpenVINO

## Get ONNX model

(tested with `torch==1.4.0`)

MaxUnpool layer in PyTorch takes two inputs - input `features` from any layer and `indices` after MaxPool layer:

```python
self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
self.unpool = nn.MaxUnpool2d(2, stride=2)

output, indices = self.pool(x)
# ...
unpooled = self.unpool(features, indices)
```

If your version of PyTorch does not support ONNX model conversion with MaxUnpool, replace every unpool layer definition
```python
self.unpool = nn.MaxUnpool2d(2, stride=2)
```
to
```python
self.unpool = Unpool2d()
```

where `Unpool2d` defined in [unpool.py](./unpool.py). Also, replace op usage from

```python
self.unpool(features, indices)
```
to
```python
self.unpool.apply(features, indices)
```

See complete example in [export_model.py](./export_model.py).

## OpenVINO Model Optimizer extension

To create OpenVINO IR, use extra `--extension` flag to specify a path to Model Optimizer extensions that perform graph transformations and register unpooling layer.

```bash
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model model_with_unpool.onnx \
    --extension mo_extensions
```

## Custom CPU extensions

You also need to build CPU extensions library which actually has Unpooling implementation:
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

net = ie.read_network('model_with_unpool.xml', 'model_with_unpool.bin')
exec_net = ie.load_network(net, 'CPU')
```
