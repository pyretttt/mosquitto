from typing import Callable, Union, TypeVar
from functools import wraps


def ascii_uppercase_prefix(len: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len]


def make_elementwise_einsum_notation() -> str:
    return "...,...->..."


def assert_dldy(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        assert kwargs["chain_jacobian"] is not None, "Multidimensional Valued function should be differentiated w.r.t scalar"
        return fn(*args, **kwargs)
    return wrapper



class Variable:
    """
    Created on binary operation (Z = X (op) Y)
    Lives in output (Z), computes gradient w.r.t. argument/s (X/Y, or both, but created for each other independently)
    Backwards whole computation graph.
    """

    def __init__(self, wrt_argument, backward_method):
        self.backward_method = backward_method
        self.wrt_argument = wrt_argument


    def backward(self, **params):
        # aka gradient of argument tensor
        jacobian_wrt_argument = self.backward_method(**params)
        assert jacobian_wrt_argument.requires_grad == False
        assert jacobian_wrt_argument.is_leaf == False
        if self.wrt_argument.is_leaf:
            # gradient accumulation
            if self.wrt_argument.grad is not None:
                self.wrt_argument.grad = self.wrt_argument.grad + jacobian_wrt_argument
            else:
                self.wrt_argument.grad = jacobian_wrt_argument
        elif self.wrt_argument.grad_fn is not None and self.wrt_argument.requires_grad:
            self.wrt_argument.grad_fn.backward(
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



def make_compound_variable(wrts_with_backwards: list):
    variables = list(filter(
        lambda x: x is not None,
        [
            (
                Variable(
                    wrt_argument=op,
                    backward_method=backward_method
                )
                if op.requires_grad and backward_method is not None
                else None
            )
            for op, backward_method in wrts_with_backwards
        ]
    ))
    if not len(variables):
        raise RuntimeError("Wrong backward chain")

    return CompoundVariable(*variables)