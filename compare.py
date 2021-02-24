import argparse
import numpy as np
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser()
parser.add_argument('-l', dest='extension', required=True, help='Path to CPU extensions library')
args = parser.parse_args()

inp = np.load('inp.npy')
ref = np.load('ref.npy')

ie = IECore()
ie.add_extension(args.extension, 'CPU')

net = ie.read_network('model.xml', 'model.bin')
net.reshape({'input': inp.shape})
exec_net = ie.load_network(net, 'CPU')

out = exec_net.infer({'input': inp})
out = next(iter(out.values()))

maxdiff = np.max(np.abs(ref - out))
print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Maximal difference:', maxdiff)
if maxdiff > 1e-5:
    exit(1)
