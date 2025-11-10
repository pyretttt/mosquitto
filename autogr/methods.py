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
                Variable(argument=op1, backward_method=make_add_backward(host=op1))
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
                Variable(op1, backward_method=make_add_backward(host=op1)),
                Variable(op2, backward_method=make_add_backward(host=op2)),
            )
            if req_grad
            else None
        ),
        is_leaf=False
    )


def make_add_backward(host: Tensor):
    def add_backward(chain_jacobian: Optional[Tensor]):
        add_partial_wrt_to_host = tensor.Tensor.diag(np.ones(shape=(len(host.data))), is_leaf=False)
        return (
            matmul(
                chain_jacobian,
                add_partial_wrt_to_host
            )
            if chain_jacobian is not None
            else add_partial_wrt_to_host
        )

    return add_backward