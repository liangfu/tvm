# pylint: disable=invalid-name, protected-access
# pylint: disable=no-member, missing-docstring, not-callable
from __future__ import absolute_import

import ctypes
from ..base import _LIB, check_call, c_str
from ..node_generic import _set_class_node_base
from .types import TVMValue, TypeCode
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func

NodeHandle = ctypes.c_void_p
__init_by_constructor__ = None

"""Maps node type to its constructor"""
NODE_TYPE = {}

def _register_node(index, cls):
    """register node class"""
    NODE_TYPE[index] = cls

def _return_node(x):
    """Return node function"""
    handle = x.v_handle
    if not isinstance(handle, NodeHandle):
        handle = NodeHandle(handle)
    tindex = ctypes.c_int()
    check_call(_LIB.TVMNodeGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = NODE_TYPE.get(tindex.value, NodeBase)
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    node = cls.__new__(cls)
    node.handle = handle
    return node


RETURN_SWITCH[TypeCode.NODE_HANDLE] = _return_node
C_TO_PY_ARG_SWITCH[TypeCode.NODE_HANDLE] = _wrap_arg_func(
    _return_node, TypeCode.NODE_HANDLE)


class NodeBase(object):
    __slots__ = ["handle"]
    # pylint: disable=no-member
    def __del__(self):
        if _LIB is not None:
            check_call(_LIB.TVMNodeFree(self.handle))

    def __getattr__(self, name):
        ret_val = TVMValue()
        ret_type_code = ctypes.c_int()
        ret_success = ctypes.c_int()
        check_call(_LIB.TVMNodeGetAttr(
            self.handle, c_str(name),
            ctypes.byref(ret_val),
            ctypes.byref(ret_type_code),
            ctypes.byref(ret_success)))
        if not ret_success.value:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (str(type(self)), name))
        return RETURN_SWITCH[ret_type_code.value](ret_val)

    def __init_handle_by_constructor__(self, fconstructor, *args):
        """Initialize the handle by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return handle is directly set into the Node object
        instead of creating a new Node.
        """
        handle = __init_by_constructor__(fconstructor, args)
        if not isinstance(handle, NodeHandle):
            handle = NodeHandle(handle)
        self.handle = handle

_set_class_node_base(NodeBase)
