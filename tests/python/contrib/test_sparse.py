import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))

import tvm

def test_tensor():
    stype = 0
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A', stype=stype)
    B = tvm.placeholder((n, l), name='B', stype=stype)
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    print(T)
    print(T.op.body)
    print(type(A))
    print(A.stype, B.stype, T.stype)
    assert(A.stype == 'csr')
    assert(T.stype == 'csr')
    assert(tuple(T.shape) == (m, n, l))
    assert(isinstance(A.op, tvm.tensor.PlaceholderOp))
    assert(A == A)
    assert(T.op.output(0) == T)
    assert(T.op.output(0).__hash__() == T.__hash__())
    d = {T.op.output(0) : 1}
    assert(d[T] == 1)
    assert(T[0][0][0].astype('float16').dtype == 'float16')

if __name__ == "__main__":
    test_tensor()
