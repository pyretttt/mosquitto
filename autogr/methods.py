from __future__ import annotations
from typing import Union, Optional
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor import Tensor
else:
    import tensor
from grad import Variable, CompoundVariable


# Reduction

def mean(op1: Tensor, dim: int):
    """Dim parameter is ignored"""
    output = np.array(op1.data.mean())
    return tensor.Tensor(
        data=output,
        requires_grad=op1.requires_grad,
        grad_fn=(
            Variable(wrt_argument=op1, backward_method=make_mean_backward(wrt_host=op1))
            if op1.requires_grad
            else None
        )
    )


def make_mean_backward(wrt_host: Tensor):
    def mean_backward(chain_jacobian: Optional[Tensor]) -> Tensor:
        mean_partial_wrt_to_host = tensor.Tensor.scalar_like(
            wrt_host.data,
            value=1.0 / wrt_host.data.size,
            is_leaf=False
        )
        dldx = (
            tensor.Tensor.from_numpy(
                np.einsum("i,...i->...i", chain_jacobian, mean_partial_wrt_to_host),
                is_leaf=False
            )
            if chain_jacobian is not None
            else mean_partial_wrt_to_host
        )
        return dldx

    return mean_backward

# Multiplication

def matmul(
    op1: Tensor,
    op2: Tensor
):
    requires_grad = op1.requires_grad or op2.requires_grad
    return tensor.Tensor(
        data=op1.data @ op2.data,
        requires_grad=requires_grad,
        grad_fn=(
            Variable(make_add_backward())
            if requires_grad
            else None
        ),
        is_leaf=False
    )

# Add

def add(
    op1: Tensor,
    op2: Union[Tensor, Number]
) -> Tensor:
    if isinstance(op2, (Number)):
        return tensor.Tensor(
            data=op1.data + op2,
            requires_grad=op1.requires_grad,
            grad_fn=(
                Variable(wrt_argument=op1, backward_method=make_add_backward(wrt_host=op1))
                if op1.requires_grad
                else None
            ),
            is_leaf=False
        )

    # both tensors
    req_grad=op1.requires_grad or op2.requires_grad
    return tensor.Tensor(
        data=op1.data + op2.data,
        requires_grad=op1.requires_grad or op2.requires_grad,
        grad_fn=(
            CompoundVariable(
                Variable(wrt_argument=op1, backward_method=make_add_backward(wrt_host=op1)),
                Variable(wrt_argument=op2, backward_method=make_add_backward(wrt_host=op2)),
            )
            if req_grad
            else None
        ),
        is_leaf=False
    )


def make_add_backward(wrt_host: Tensor):
    """
    Maps op1.shape -> op1.shape
    So jacobian has size (op1.shape, op1.shape)
    """
    def add_backward(chain_jacobian: Optional[Tensor]):
        add_partial_wrt_to_host = tensor.Tensor.from_numpy(np.ones_like(wrt_host.data), is_leaf=False)
        dldx = (
            tensor.Tensor.from_numpy(
                # elementwise multiplication with ones, can actually be just chain_jacobian.data
                data=np.einsum("...,...->...", chain_jacobian.data, add_partial_wrt_to_host.data),
                is_leaf=False
            )
            if chain_jacobian is not None
            else add_partial_wrt_to_host
        )
        return dldx

    return add_backward