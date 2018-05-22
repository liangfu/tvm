"""
MULTIBOX_PRIOR Operator
====================
Multibox_Prior operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from .. import util
from .. import transform

@tvm.target.generic_func
def multibox_prior(data, sizes_str, ratios_str):
    """Multibox_Prior forward operators.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    sizese : float
        Size values for multibox

    ratios : float
        Ratio values for multibox

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, c_in, h_in, w_in = util.get_const_tuple(data.shape)
    sizes = util.get_const_float_tuple(sizes_str)
    ratios = util.get_const_float_tuple(ratios_str)
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    num_anchors = num_sizes - 1 + num_ratios
    print(sizes_str, ratios_str, str(num_anchors))
    out = tvm.compute((num_anchors * h_in * w_in, 4), lambda j, i: 0.0, tag='multibox_prior')
    return transform.reshape(out, (num_anchors * h_in * w_in, 4))

