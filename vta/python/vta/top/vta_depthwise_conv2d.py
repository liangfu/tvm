# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Depthwise conv2D operator declaration and schedule registration for VTA."""

import numpy as np

import tvm
from tvm import autotvm
import topi

from ..environment import get_env

@autotvm.register_topi_compute(topi.nn.depthwise_conv2d_nchw, 'vta', 'direct')
def packed_depthwise_conv2d(cfg,
                        data,
                        kernel,
                        strides,
                        padding,
                        dilation,
                        out_dtype):
    """ Packed depthwise conv2d nchw function."""
    assert dilation == (1, 1)
    kp = 1

    if padding[0]:
        pad_data = topi.nn.pad(data,
                               [0, 0, padding[0], padding[1], 0, 0],
                               [0, 0, padding[0]+kp, padding[1]+kp, 0, 0], name="pad_data")
    else:
        pad_data = data

    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    assert data.dtype == "int8", data.dtype
    assert kernel.dtype == "int8", kernel.dtype
    assert out_dtype == "int32", out_dtype

    ishape = topi.util.get_const_tuple(data.shape)
    kshape = topi.util.get_const_tuple(kernel.shape)
    multiplier = kshape[1]
    blocksize = ishape[-1]

    assert blocksize == 1
    assert kshape[2] == 3 and kshape[3] == 3

    pad_kernel = topi.nn.pad(kernel, [0,0,0,0,0,0], [0,0,kp,kp,0,0], name="pad_kernel")
    pad_kernel = topi.reshape(pad_kernel, (kshape[0], kshape[1], 1, 16, 1, 1))

    oheight = topi.util.get_const_int((pad_data.shape[2] - 4) // strides[0] + 1)
    owidth = topi.util.get_const_int((pad_data.shape[3] - 4) // strides[1] + 1)
    oshape = (ishape[0], ishape[1] * multiplier, oheight, owidth, ishape[4], ishape[5])

    print("--")
    print("ishape={}, kshape={}, oshape={}".format(ishape, kshape, oshape))
    # print(topi.reshape)
    # print(topi.transpose)
    # print(topi.nn.pad)
    print("--")

    ko = tvm.reduce_axis((0, 1), name='ko') # kernel
    ki = tvm.reduce_axis((0, 16), name='ki')
    hstride, wstride = strides
    out = tvm.compute(
        oshape,
        lambda bo, co, i, j, bi, ci: tvm.sum(
            pad_data[bo,
                     co,
                     i * hstride + tvm.indexdiv(ki, 4),
                     j * wstride + tvm.indexmod(ki, 4),
                     bi,
                     ci].astype(out_dtype) *
            pad_kernel[co,
                   0,
                   ko,
                   ki,
                   ci,
                   0].astype(out_dtype),
            axis=[ko, ki]),
        name="res", tag="packed_depthwise_conv2d")

    cfg.add_flop(np.prod(topi.util.get_const_tuple(oshape)) * 16)

    return out


@autotvm.register_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw, 'vta', 'direct')
def schedule_packed_depthwise_conv2d(cfg, outs):
    """ Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

    def _traverse(op):
        print(op.output(0).op.name, [tensor.op.name for tensor in op.input_tensors], op.tag)
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_depthwise_conv2d"
            conv2d_res.append(op)
            print('--')

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    s = tvm.create_schedule(output.op)

    ##### space definition begin #####
    b, co, xi, xj, _, _ = s[conv2d_stage].op.axis
    ko, ki = s[conv2d_stage].op.reduce_axis
    cfg.define_split('tile_h', xi, num_outputs=2)
    cfg.define_split('tile_w', xj, num_outputs=2)
    cfg.define_split('tile_co', co, num_outputs=2)
    ###### space definition end ######

    if cfg.is_fallback:
        cfg.fallback_split('tile_h', [-1, 16])
        cfg.fallback_split('tile_w', [-1, 1])
        cfg.fallback_split('tile_co', [-1, 1])

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    
    env = get_env()
    
    # setup pad
    # TODO(liangfu): cache read pad_data into wgt_scope
    # if pad_data is not None:
    #     cdata = pad_data
    #     s[pad_data].set_scope(env.wgt_scope)
    #     cdata = s.cache_read(pad_data, env.wgt_scope, [conv2d_stage])
    # else:
    #     cdata = s.cache_read(data, env.wgt_scope, [conv2d_stage])
    cdata = s.cache_read(pad_data, env.wgt_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.inp_scope, [conv2d_stage])
    # s[conv2d_stage].set_scope(env.acc_scope)
    
    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))
    
    # set ewise scope
    # for op in ewise_ops:
    #     s[op].set_scope(env.acc_scope)
    #     s[op].pragma(s[op].op.axis[0], env.alu)
    for op in const_ops:
        s[op].compute_inline()
    
    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg['tile_co'].apply(s, output, x_co)
    x_i0, x_i1 = cfg['tile_h'].apply(s, output, x_i)
    x_j0, x_j1 = cfg['tile_w'].apply(s, output, x_j)
    s[output].reorder(x_bo, x_co0, x_i0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0
    
    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)
    
    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)
    
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    ko, ki = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, ko, x_j, x_co, x_i, x_bi, x_ci, ki)

    # ko, _ = cfg['tile_w'].apply(s, conv2d_stage, ko)
    s[cdata].compute_at(s[conv2d_stage], x_i)
    s[ckernel].compute_at(s[conv2d_stage], ko)

    # fused = s[cdata].fuse(*list(s[cdata].op.axis))
    
    # Use VTA instructions
    # s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    # s[conv2d_stage].tensorize(x_bi, env.gemm)
    # s[output].pragma(x_co1, env.dma_copy)

    return s
