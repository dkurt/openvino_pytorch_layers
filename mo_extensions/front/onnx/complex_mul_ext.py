from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from ...ops.ComplexMultiplication import ComplexMultiplication

class ComplexMultiplicationExtractor(FrontExtractorOp):
    op = 'ComplexMultiplication'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'is_conj': onnx_attr(node, 'is_conj', 'i'),
        }
        ComplexMultiplication.update_node_stat(node, attrs)
        return cls.enabled
