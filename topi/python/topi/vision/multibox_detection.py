"""
MULTIBOX_DETECTION Operator
====================
Multibox_Detection operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from .. import util
from .. import transform

@tvm.target.generic_func
def multibox_detection(cls_prob, loc_pred, anchor):
    """Multibox_Detection forward operators.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    cls_prob : tvm.Tensor
        Class probability tensor for multibox_detection

    loc_pred : tvm.Tensor
        Location predicts tensor for multibox_detection

    anchor : tvm.Tensor
        Anchors tensor for multibox_detection

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    cshape = cls_prob.shape
    lshape = loc_pred.shape
    ashape = anchor.shape
    oshape = (cshape[0], ashape[1], 7)
    out = tvm.compute((cshape[0], ashape[1], 7), lambda k, j, i: 0.0, tag="multibox_detection")
    return transform.reshape(out, (cshape[0], ashape[1], 7))
