# mo_extensions/ops/MaxPoolGrad.py
import numpy as np
from mo.graph.graph import Node, Graph
from mo.ops.op import Op

def shape_infer(node):
    # Inputs: [max_pool_input, max_pool_output, unpool_input]
    node.out_node(0).shape = node.in_node(0).shape
    if 'output_size' in node:
        node.out_node(0).shape[2] = node['output_size'][2]
        node.out_node(0).shape[3] = node['output_size'][3]

class MaxPoolGrad(Op):
    op = 'MaxPoolGrad'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': shape_infer
        }, attrs)
