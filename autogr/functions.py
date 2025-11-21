from typing import Optional

import numpy as np
from tensor import Tensor
from grad import make_compound_variable, Variable, make_elementwise_einsum_notation, assert_dldy

# Sin

def sin(tensor: Tensor, /) -> Tensor:
    return Tensor(
        data=np.sin(tensor.data),
        requires_grad=tensor.requires_grad,
        grad_fn=(
            Variable(
                wrt_argument=tensor,
                backward_method=make_sin_backward(wrt_argument=tensor)
            )
            if tensor.requires_grad
            else None
        ),
        is_leaf=False
    )


def make_sin_backward(wrt_argument: Tensor):
    @assert_dldy
    def sin_backward(chain_jacobian: Optional[Tensor]) -> Tensor:
        return Tensor.from_numpy(
            data=chain_jacobian.data * np.cos(wrt_argument.data),
            is_leaf=False
        )

    return sin_backward


# ReLU
def relu(tensor: Tensor, /) -> Tensor:
    mask = np.zeros_like(tensor.data)
    mask[tensor.data >= 0] = 1
    return Tensor(
        data=np.maximum(tensor.data, 0),
        requires_grad=tensor.requires_grad,
        grad_fn=(
            Variable(
                wrt_argument=tensor,
                backward_method=make_relu_backward(wrt_argument=tensor, mask=mask)
            )
            if tensor.requires_grad
            else None
        ),
        is_leaf=False
    )


def make_relu_backward(wrt_argument: Tensor, mask: np.ndarray):
    @assert_dldy
    def relu_backward(chain_jacobian: Optional[Tensor]):
        return Tensor.from_numpy(
            data=chain_jacobian.data * mask,
            is_leaf=False
        )

    return relu_backward