from __future__ import absolute_import
import math
import numpy as np

from . import _quantize
from .. import expr as _expr
from .. import ir_pass as _ir_pass
from .. import build_module as _build
from .. import op as _op
from ... import make as _make
from ..base import NodeBase, register_relay_node


class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3


@register_relay_node("relay.quantize.QConfig")
class QConfig(NodeBase):
    """Configure the quantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use qconfig instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "global_scale": 8.0,
        "skip_k_conv": 1,
        "round_for_shift": True,
        "store_lowbit_output": True,
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(QConfig, self).__init__(handle)
        self.handle = handle

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope(self)

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(QConfig, self).__setattr__(name, value)


def current_qconfig():
    """Get the current quantization configuration."""
    return _quantize._GetCurrentQConfig()


def qconfig(**kwargs):
    """Configure the quantization behavior by setting config variables.

    Parameters
    ---------
    nbit_dict: dict of QAnnotateKind -> int
        Number of bit for every kind of annotate field.

    global_scale: float
        The global scale for calibration.

    skip_k_conv: int
        The number of skipped conv2d.

    round_for_shift: boolean
        Whether to add bias for rounding during shift.

    store_lowbit_output: boolean
        Whether to store low-bit integer back as output before dequantizing.
        Some accelerators need this, e.g. VTA.

    Returns
    -------
    config: QConfig
        The quantization configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in QConfig._node_defaults.items()}
    config = _make.node("relay.quantize.QConfig", **node_args)
    return config


CONV_COUNTER = 0


def _conv_counter():
    """Get the global counter for conv2d."""
    return CONV_COUNTER


def _set_conv_counter(n):
    """Set the value of the global conv2d counter."""
    global CONV_COUNTER
    CONV_COUNTER = n


def annotate(graph):
    """Given a float32 graph, annotate will rewrite the graph
    and return back a graph which simulates the error brought by
    current quantization scheme.

    Parameters
    ---------
    graph: Function
        The original graph
    """
    _set_conv_counter(0)  # reset counter
    return _quantize.annotate(graph)


def calibrate(graph, dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    graph: Function
        The simulation graph after annotation.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.
    """
    def power2_scale(arr):
        """calculate weight scale with nearest mode-2 scale"""
        val = np.amax(np.abs(arr.asnumpy()))
        if val == 0.0:
            return 1.0
        else:
            return 2**math.ceil(math.log(val, 2))

    cfg = current_qconfig()
    const_params = {}
    quantize_op = _op.get("simulated_quantize")

    def visit_func(e):
        if isinstance(e, _expr.Call) and e.op == quantize_op:
            ndata, ndom_scale, nclip_min, nclip_max = e.args
            attrs = e.attrs
            kind = attrs.kind
            TABLE = [0, cfg.nbit_input, cfg.nbit_weight, cfg.nbit_activation]
            nbit = TABLE[kind]

            valid_bit = nbit
            if attrs.sign:
                valid_bit = nbit - 1

            if kind == QAnnotateKind.WEIGHT:
                var = e.args[0]
                assert isinstance(var, _expr.Constant)
                arr = var.data
                scale = power2_scale(arr)
            else:
                scale = cfg.global_scale

            const_params[ndom_scale] =  \
                _expr.const(scale/2**valid_bit, 'float32')
            const_params[nclip_min] = _expr.const(-(2**valid_bit-1), 'float32')
            const_params[nclip_max] = _expr.const((2**valid_bit-1), 'float32')
        return

    _ir_pass.post_order_visit(graph, visit_func)
    f = _expr.bind(graph, const_params)
    return f


def realize(graph):
    """The realize pass will transform the simulated quantized
    graph, which computes with float32 actually, to a real low-bit
    integer graph. It will replace the simulated_quantize with
    several fine-grained operators like add, multiply, and shift
    as more as possible for performance (fusion, etc.)

    Parameters
    ---------
    graph: Function
        The simulated graph after calibrating.
    """
    return _quantize.realize(graph)


def quantize(graph, params=None, dataset=None):
    """ The quantization procedure.

    Parameters
    ---------
    graph: Function
        The original graph.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.
    """
    with _build.build_config(opt_level=3):
        graph = _build.optimize(graph, params)
    graph = annotate(graph)
    graph = calibrate(graph, dataset)
    graph = realize(graph)
    with _build.build_config(opt_level=3):
        graph = _build.optimize(graph)
    return graph
