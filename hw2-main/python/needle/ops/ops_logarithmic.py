from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    """
    Not only solve the overflow problem but also solve the under flow problem.
    Although its name is `LogSoftmax` but actually it use log for p(hi):
    exp(hi) / sum(j -> k, exp(hj))
    ==> log(exp(hi - C) / sum(j -> k, exp(hj - C)))
    ==> (hi - C) - log(sum(j -> k, exp(hj - C)))
    1. minus C to avoid EXP operation overflow
    2. transform equation and make sure sum(j -> k) >= 1 at least to avoid log(0) this underflow
    """
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    """
    This is the child-module of softmax loss computation.
    And for `EXP` computation stability, replace Z with Z - max(Z) + max(Z) to solve data overflow.
    """
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        """
        Args:
            Z: 2D NDArray like [Batch size, num_classes]

        Returns: stable computation result of LogSumExp
        """
        """
        The `max_z_original` and `max_z_reduce` is same at data but one is 2D shape for Z and another is 1D shape
        for logsumexp's result.
        """
        max_z_original = array_api.max(Z, self.axes, keepdims=True)
        max_z_reduce = array_api.max(Z, self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)) + max_z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

