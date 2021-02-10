# mo_extensions/front/onnx/max_unpool.py
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from extensions.ops.upsample import UpsampleOp
from extensions.ops.activation_ops import Abs
from extensions.ops.elementwise import Sub, Less, Mul
from mo.ops.const import Const
from extensions.ops.Cast import Cast
from mo.front.onnx.extractors.utils import onnx_attr

class MaxUnpool(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('max_pool0', dict(op='MaxPool')),
                ('max_pool1', dict(op='MaxPool')),
                ('slice', dict(op='AttributedSlice')),
                ('sub', dict(op='Sub')),
                ('unpool', dict(op='Unpooling')),
            ],
            edges=[
                ('max_pool1', 'slice'),
                ('max_pool0', 'sub', {'in': 0}),
                ('slice', 'sub', {'in': 1}),
                ('sub', 'unpool', {'in': 1}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        max_pool = match['max_pool0']
        max_pool_input = max_pool.in_port(0).get_source().node
        unpool = match['unpool']
        unpool_input = unpool.in_port(0).get_source().node

        max_pool.out_port(1).disconnect()

        # Resize Pooling's output
        pool_out_resized = UpsampleOp(graph, dict(name=max_pool.name + '/resize',
                                                  height_scale=2.0,
                                                  width_scale=2.0,
                                                  mode='nearest')).create_node([max_pool])
        x_resized = UpsampleOp(graph, dict(name=unpool_input.name + '/resize',
                                           height_scale=2.0,
                                           width_scale=2.0,
                                           mode='nearest')).create_node([unpool_input])
        diff = Sub(graph, dict(name=unpool.name + '/sub')).create_node([pool_out_resized, max_pool_input])
        abs = Abs(graph, dict(name=unpool.name + '/abs')).create_node([diff])
        thresh = Const(graph, {'value': 1e-6}).create_node()
        less = Less(graph, dict(name=unpool.name + '/less')).create_node([abs, thresh])
        less = Cast(graph, dict(name=unpool.name + '/cast', dst_type=np.float32)).create_node([less])
        res = Mul(graph, dict(name=unpool.name + '/mul')).create_node([x_resized, less])

        unpool.out_port(0).get_connection().set_source(res.out_port(0))
