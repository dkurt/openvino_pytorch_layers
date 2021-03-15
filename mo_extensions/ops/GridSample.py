import numpy as np
from mo.graph.graph import Node, Graph
from mo.ops.op import Op

def shape_infer(node):
    # Inputs: [x, grid]
    # Grid has a shape NxHxWx2
    assert(len(node.in_nodes()) == 2)
    node.out_node(0).shape = node.in_node(0).shape  # NC
    node.out_node(0).shape[2] = node.in_node(1).shape[1]  # H
    node.out_node(0).shape[3] = node.in_node(1).shape[2]  # W

class GridSample(Op):
    op = 'GridSample'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': shape_infer
        }, attrs)
