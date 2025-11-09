from typing import Callable, Union, TypeVar


class Variable:
    """
    Created on binary operation (Z = X \cdot Y)
    Lives in output (Z), computes gradient w.r.t. argument/s (X/Y, or both, but created for each other independently)
    Backwards whole computation graph.
    """

    def __init__(self, argument, backward_method):
        self.backward_method = backward_method
        self.argument = argument


    def backward(self, **params):
        jacobian_wrt_argument = self.backward_method(**params)
        if self.argument.is_leaf:
            self.argument.grad = jacobian_wrt_argument
        elif self.argument.grad_fn is not None and self.argument.requires_grad:
            self.argument.grad_fn.backward_method(
                chain_jacobian=jacobian_wrt_argument
            )
        else:
            raise "Malfunctioned non leaf tensor"