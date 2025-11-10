from typing import Callable, Union, TypeVar


class Variable:
    """
    Created on binary operation (Z = X (op) Y)
    Lives in output (Z), computes gradient w.r.t. argument/s (X/Y, or both, but created for each other independently)
    Backwards whole computation graph.
    """

    def __init__(self, argument, backward_method):
        self.backward_method = backward_method
        self.argument = argument


    def backward(self, **params):
        # aka gradient of argument tensor
        jacobian_wrt_argument = self.backward_method(**params)
        assert jacobian_wrt_argument.requires_grad == False
        assert jacobian_wrt_argument.is_leaf == False
        if self.argument.is_leaf:
            # gradient accumulation
            if self.argument.grad is not None:
                self.argument.grad = self.argument.grad + jacobian_wrt_argument
            else:
                self.argument.grad = jacobian_wrt_argument
        elif self.argument.grad_fn is not None and self.argument.requires_grad:
            self.argument.grad_fn.backward(
                chain_jacobian=jacobian_wrt_argument
            )
        else:
            raise "Malfunctioned non leaf tensor"


class CompoundVariable:
    """
    If operation contains multiple hosts with requires_grad,
    then compound variables i-multiplexes (inverse multiplex) chain jacobian to each host(argument)
    """
    def __init__(self, *variables):
        self.variables = variables


    def backward(self, **params):
        for variable in self.variables:
            variable.backward(**params)