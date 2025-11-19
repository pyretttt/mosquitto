from __future__ import annotations
from typing import Union, Optional
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor import Tensor
else:
    import tensor
from grad import Variable, make_compound_variable, make_elementwise_einsum_notation, ascii_uppercase_prefix, assert_dldy

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
        ),
        is_leaf=False
    )


def make_mean_backward(wrt_host: Tensor):
    def mean_backward(chain_jacobian: Optional[Tensor]) -> Tensor:
        mean_partial_wrt_to_host = tensor.Tensor.scalar_like(
            wrt_host.data,
            value=1.0 / wrt_host.data.size,
            is_leaf=False
        )
        dldx = (
            # chain jacobian is just scalar w.r.t. L
            tensor.Tensor.from_numpy(
                mean_partial_wrt_to_host.data * chain_jacobian.data,
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
    """
    Broadcasted over `...` for shapes `...ij`, `...jk`
    """
    assert op1.dim() >= op2.dim(), "Not supported"
    requires_grad = op1.requires_grad or op2.requires_grad
    return tensor.Tensor(
        data=np.matmul(op1.data, op2.data),
        requires_grad=requires_grad,
        grad_fn=(
            make_compound_variable(
                zip(
                    [op1, op2],
                    [make_matmul_backward(wrt_host=op1, op2=op2, left_multiply=True), make_matmul_backward(wrt_host=op2, op2=op1, left_multiply=False)]
                )
            )
            if requires_grad
            else None
        ),
        is_leaf=False
    )

def make_matmul_backward(wrt_host: Tensor, op2: Tensor, left_multiply: bool):
    @assert_dldy
    def matmul_backward(chain_jacobian: Optional[Tensor]):
        if left_multiply:
            # For now I assume that it's ...jk data matrix
            return tensor.Tensor.from_numpy(
                # Should work for most broadcasting's
                data=np.einsum("...ij,...jk->...ik", chain_jacobian.data, op2.data.T),
                is_leaf=False
            )
        else:
            einsum_prefix_for_summ_indices = ascii_uppercase_prefix(len=op2.dim() - wrt_host.dim())
            # For now I assume that it's 2x2 matrix weights
            return tensor.Tensor.from_numpy(
                # Should work for most broadcasting's
                data=np.einsum(
                    f"{einsum_prefix_for_summ_indices}pj,{einsum_prefix_for_summ_indices}jq->pq",
                    np.swapaxes(op2.data, -1, -2),
                    chain_jacobian.data
                ),
                is_leaf=False
            )

    return matmul_backward


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
            make_compound_variable(
                zip(
                    [op1, op2],
                    [make_add_backward(wrt_host=op1), make_add_backward(wrt_host=op2)]
                )
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
    @assert_dldy
    def add_backward(chain_jacobian: Optional[Tensor]) -> Tensor:
        add_partial_wrt_to_host = tensor.Tensor.from_numpy(np.ones_like(wrt_host.data), is_leaf=False)
        dldx = tensor.Tensor.from_numpy(
            # elementwise multiplication with ones, can actually be just chain_jacobian.data
            data=np.einsum(make_elementwise_einsum_notation(), chain_jacobian.data, add_partial_wrt_to_host.data),
            is_leaf=False
        )
        return dldx

    return add_backward