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
    return _expr.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


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
    cnt = _conv_counter()
    if cnt < current_qconfig().skip_k_conv:
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
    if _conv_counter() <= current_qconfig().skip_k_conv:
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


def _get_expr_kind(anno):
    if isinstance(anno, QAnnotateExpr):
        return anno.expr, anno.kind
    else:
        return anno, None


@register_annotate_function("add")
def add_rewrite(ref_call, new_args, ctx):
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None
    elif lhs_kind is None and rhs_kind is not None:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
    elif lhs_kind is not None and rhs_kind is None:
        rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
    else:
        pass

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


def identity_rewrite(ref_call, new_args, ctx):
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None
    else:
        ret_expr = _forward_op(ref_call, [x_expr])
        return QAnnotateExpr(ret_expr, x_kind)


register_annotate_function("nn.relu", identity_rewrite)
