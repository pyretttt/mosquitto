from typing import Union, Optional
from numbers import Number
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensor import Tensor
else:
    import tensor

from grad import Variable
    

def matmul(
    op1: Tensor,
    op2: Tensor
):
    requires_grad = op1.requires_grad or op2.requires_grad
    return Tensor(
        data=op1.data @ op2.data,
        requires_grad=requires_grad,
        grad_fn=(
            Variable(make_add_backward())
            if op1.requires_grad 
            else None
        )
    )

# Add

def add(
    op1: Tensor, 
    op2: Union[Tensor, Number]
) -> Tensor:
    if isinstance(op2, (Number)):
        return Tensor(
            data=op1.data + op2, 
            requires_grad=op1.requires_grad,
            grad_fn=(
                Variable(make_add_backward())
                if op1.requires_grad 
                else None
            )
        )
        # if op1.requires_grad:
            


def make_add_backward(host: Tensor):
    def add_backward(chain_jacobian: Optional[Tensor]):
        return (
            matmul(
                chain_jacobian, 
                tensor.Tensor.ones_like(host)
            )
            if chain_jacobian is not None
            else tensor.Tensor.ones_like(host)
        )

    return add_backward