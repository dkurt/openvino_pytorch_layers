import numpy as np
from openvino.inference_engine import IECore

inp = np.load('inp.npy')
ref = np.load('ref.npy')

ie = IECore()
ie.add_extension('user_ie_extensions/build/libuser_cpu_extension.so', 'CPU')

net = ie.read_network('model_with_unpool.xml', 'model_with_unpool.bin')
net.reshape({'input': inp.shape})
exec_net = ie.load_network(net, 'CPU')

out = exec_net.infer({'input': inp})
out = next(iter(out.values()))

maxdiff = np.max(np.abs(ref - out))
print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Maximal difference:', maxdiff)
if maxdiff > 1e-6:
    exit(1)
