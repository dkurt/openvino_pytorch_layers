# from mo.front.extractor import FrontExtractorOp
# from mo.front.onnx.extractors.utils import onnx_attr
# from ...ops.GridSample import GridSample

# class GridSampleFrontExtractor(FrontExtractorOp):
#     op = 'GridSample'
#     enabled = True

#     @classmethod
#     def extract(cls, node):
#         GridSample.update_node_stat(node)
#         return cls.enabled
