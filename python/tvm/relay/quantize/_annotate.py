from __future__ import absolute_import

from . import _quantize
from .quantize import QAnnotateKind, current_qconfig
from .quantize import _conv_counter, _set_conv_counter
from .. import expr as _expr
from .. import op as _op
from ..base import register_relay_node
from ..._ffi.function import register_func


@register_relay_node
class QAnnotateExpr(_expr.TempExpr):
    """A special kind of Expr for Annotating.

    Parameters
    ---------
    expr: Expr
        the original relay ir expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """
    def __init__(self, expr, kind):
        self.__init_handle_by_constructor__(
            _quantize.make_annotate_expr, expr, kind)


def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return _expr.Call(ref_call.op, args,
                      ref_call.attrs, ref_call.type_args)


def register_annotate_function(op_name, frewrite=None, level=10):
    """register a rewrite function for operator, used by annotation.

    Parameters
    ---------
    op_name: str
        The name of operation

    frewrite : function, optional
        The function to be registered.

    level : int, optional
        The priority level
    """
    return _op.register(op_name, "FQAnnotateRewrite", frewrite, level)


@register_func("relay.quantize.attach_simulated_quantize")
def attach_simulated_quantize(data, kind):
    """Attach a simulated quantize operation after input data expr.

    Parameters
    ---------
    data: Expr
        the original data expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """
    dom_scale = _expr.var("dom_scale")
    clip_min = _expr.var("clip_min")
    clip_max = _expr.var("clip_max")
    return _quantize.simulated_quantize(data, dom_scale, clip_min,
                                        clip_max, True, "round", kind)


@register_annotate_function("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    cfg = current_qconfig()
    cnt = _conv_counter()
    if cnt < cfg.skip_k_conv:
        _set_conv_counter(cnt + 1)
        return None
    _set_conv_counter(cnt + 1)

    lhs, rhs = new_args
    if isinstance(lhs, QAnnotateExpr):
        lhs_expr = lhs.expr
        if lhs.kind != QAnnotateKind.INPUT:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
    else:
        lhs_expr = attach_simulated_quantize(lhs, QAnnotateKind.INPUT)

    assert not isinstance(rhs, QAnnotateExpr)
    rhs_expr = attach_simulated_quantize(rhs, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("multiply")
def multiply_rewrite(ref_call, new_args, ctx):
    cfg = current_qconfig()
    if _conv_counter() <= cfg.skip_k_conv:
        return None

    lhs, rhs = new_args
    if not isinstance(lhs, QAnnotateExpr) and not isinstance(rhs, QAnnotateExpr):
        return None
    elif lhs.kind == QAnnotateKind.ACTIVATION and not isinstance(rhs, QAnnotateExpr):
        lhs_expr = attach_simulated_quantize(lhs.expr, QAnnotateKind.INPUT)
        rhs_expr = attach_simulated_quantize(rhs, QAnnotateKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    else:
        raise ValueError


@register_annotate_function("add")
def add_rewrite(ref_call, new_args, ctx):
    cfg = current_qconfig()
    if _conv_counter() <= cfg.skip_k_conv:
        return None

    lhs, rhs = new_args
    if not isinstance(lhs, QAnnotateExpr) and not isinstance(rhs, QAnnotateExpr):
        # on float domain
        return None
    elif not isinstance(lhs, QAnnotateExpr) and rhs.kind == QAnnotateKind.ACTIVATION:
        # addition for residual, but lhs are calculated on real domain
        lhs_expr = attach_simulated_quantize(lhs, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs.expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    elif lhs.kind == QAnnotateKind.INPUT and not isinstance(rhs, QAnnotateExpr):
        # TODO ?
        rhs_expr = attach_simulated_quantize(rhs, QAnnotateKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs.expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    elif lhs.kind == QAnnotateKind.ACTIVATION and not isinstance(rhs, QAnnotateExpr):
        # the most common situation, e.g. bias add in bn
        rhs_expr = attach_simulated_quantize(rhs, QAnnotateKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs.expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    elif lhs.kind == QAnnotateKind.INPUT and rhs.kind == QAnnotateKind.ACTIVATION:
        # addition for residual, but lhs are muti-refered
        expr = _forward_op(ref_call, [lhs.expr, rhs.expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    elif lhs.kind == QAnnotateKind.ACTIVATION and rhs.kind == QAnnotateKind.INPUT:
        # TODO ?
        expr = _forward_op(ref_call, [lhs.expr, rhs.expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    elif lhs.kind == QAnnotateKind.ACTIVATION and rhs.kind == QAnnotateKind.ACTIVATION:
        # addition for residual
        expr = _forward_op(ref_call, [lhs.expr, rhs.expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    else:
        raise ValueError


@register_annotate_function("nn.relu")
def relu_rewrite(ref_call, new_args, ctx):
    cfg = current_qconfig()
    if _conv_counter() <= cfg.skip_k_conv:
        return None

    x = new_args[0]
    if isinstance(x, QAnnotateExpr):
        expr = _forward_op(ref_call, [x.expr])
        return QAnnotateExpr(expr, x.kind)
    else:
        return None
