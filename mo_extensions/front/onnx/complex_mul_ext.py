from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from ...ops.ComplexMultiplication import ComplexMultiplication


class ComplexMultiplicationExtractor(FrontExtractorOp):
    op = "ComplexMultiplication"
    enabled = True

    @classmethod
    def extract(cls, node):
        ComplexMultiplication.update_node_stat(node)
        return cls.enabled
