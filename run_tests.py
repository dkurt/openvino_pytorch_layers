# NOTE: import order is critical for now: extensions, openvino and only then numpy
from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

class TestLayers(unittest.TestCase):
    def convert_model(self):
        subprocess.run([sys.executable,
                        '-m',
                        'mo',
                        '--input_model=model.onnx',
                        '--extension', Path(__file__).absolute().parent / 'mo_extensions'],
                       check=True)

    def run_test(self, convert_ir=True, test_onnx=False, num_inputs=1, threshold=1e-5):
        if convert_ir and not test_onnx:
            self.convert_model()

        inputs = {}
        shapes = {}
        for i in range(num_inputs):
            suffix = '{}'.format(i if i > 0 else '')
            data = np.load('inp' + suffix + '.npy')
            inputs['input' + suffix] = data
            shapes['input' + suffix] = data.shape

        ref = np.load('ref.npy')

        ie = IECore()
        ie.add_extension(get_extensions_path(), 'CPU')
        ie.set_config({'CONFIG_FILE': 'user_ie_extensions/gpu_extensions.xml'}, 'GPU')

        net = ie.read_network('model.onnx' if test_onnx else 'model.xml')
        net.reshape(shapes)
        exec_net = ie.load_network(net, 'CPU')

        out = exec_net.infer(inputs)
        out = next(iter(out.values()))

        diff = np.max(np.abs(ref - out))
        self.assertLessEqual(diff, threshold)


    def test_unpool(self):
        from examples.unpool.export_model import export
        export(mode='default')
        self.run_test()


    def test_unpool_reshape(self):
        from examples.unpool.export_model import export
        export(mode='dynamic_size', shape=[5, 3, 6, 9])
        self.run_test()

        export(mode='dynamic_size', shape=[4, 3, 17, 8])
        self.run_test(convert_ir=False)


    def test_fft(self):
        from examples.fft.export_model import export

        for shape in [[5, 120, 2], [4, 240, 320, 2], [3, 5, 240, 320, 2]]:
            export(shape=shape)
            self.run_test()


    def test_fft_roll(self):
        from examples.fft.export_model_with_roll import export

        export()
        self.run_test()
        self.run_test(test_onnx=True)


    def test_grid_sample(self):
        from examples.grid_sample.export_model import export

        export()
        self.run_test(num_inputs=2)
        self.run_test(num_inputs=2, test_onnx=True)


    def test_complex_mul(self):
        from examples.complex_mul.export_model import export

        for shape in [[3, 2, 4, 8, 2], [3, 1, 4, 8, 2]]:
            export(other_shape=shape)
            self.run_test(num_inputs=2)
            self.run_test(num_inputs=2, test_onnx=True)


    def test_deformable_conv(self):
        from examples.deformable_conv.export_model import export
        export()
        self.run_test()
        self.run_test(test_onnx=True)


if __name__ == '__main__':
    unittest.main()
