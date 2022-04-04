from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


def shape_infer(node):
    assert len(node.in_nodes()) == 2
    node.out_node(0).shape = node.in_node(0).shape


class ComplexMultiplication(Op):
    op = "ComplexMultiplication"
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(
            graph,
            {
                "type": __class__.op,
                "op": __class__.op,
                "in_ports_count": 2,
                "out_ports_count": 1,
                "infer": shape_infer,
            },
            attrs,
        )
