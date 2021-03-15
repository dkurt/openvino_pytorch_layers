import argparse
import numpy as np
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser(description='Compare OpenVINO implementation with reference data')
parser.add_argument('--num_inputs', type=int, default=1)
args = parser.parse_args()

inputs = {}
shapes = {}
for i in range(args.num_inputs):
    suffix = '{}'.format(i if i > 0 else '')
    data = np.load('inp' + suffix + '.npy')
    inputs['input' + suffix] = data
    shapes['input' + suffix] = data.shape

ref = np.load('ref.npy')

ie = IECore()
ie.add_extension('user_ie_extensions/build/libuser_cpu_extension.so', 'CPU')

net = ie.read_network('model.xml', 'model.bin')
net.reshape(shapes)
exec_net = ie.load_network(net, 'CPU')

out = exec_net.infer(inputs)
out = next(iter(out.values()))

# print(inputs['input'])
# print(inputs['input1'])
# print(ref)
# print(out)

maxdiff = np.max(np.abs(ref - out))
print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Maximal difference:', maxdiff)
if maxdiff > 1e-5:
    exit(1)
