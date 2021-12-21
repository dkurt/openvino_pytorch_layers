#!/usr/bin/env python
import os
import sys
from setuptools import setup

if not 'VERSION' in os.environ:
    raise Exception('Specify package version by <VERSION> environment variable')

if not 'EXT_LIB' in os.environ:
    raise Exception('Specify <EXT_LIB> environment variable with a path to extensions library')

if sys.platform == 'win32':
    path_to_onnx = os.path.join(os.environ["INTEL_OPENVINO_DIR"], "deployment_tools\\ngraph\\lib\\onnx_importer.dll")
elif sys.platform == 'linux':
    path_to_onnx = os.path.join(os.environ["INTEL_OPENVINO_DIR"], "deployment_tools/ngraph/lib/libonnx_importer.so")
else:
    path_to_onnx = os.path.join(os.environ["INTEL_OPENVINO_DIR"], "deployment_tools/ngraph/lib/libonnx_importer.dylib")

setup(name='openvino-extensions',
      version=os.environ['VERSION'],
      author='Dmitry Kurtaev',
      url='https://github.com/dkurt/openvino_pytorch_layers',
      packages=['openvino_extensions'],
      data_files=[('../../openvino_extensions', [os.environ['EXT_LIB'], path_to_onnx])],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
      ],
)
