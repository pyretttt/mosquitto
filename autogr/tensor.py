from typing import List, Optional, Self, Union
import numpy as np
import numbers

from grad import Variable


class Tensor():
    def __init__(
        self,
        data: np.array,
        requires_grad: bool,
        grad_fn: Optional[Variable] = None,
        is_leaf: bool = True
    ):
        self.data = data
        self.requires_grad = requires_grad
        self._grad_fn = grad_fn
        self.is_leaf = is_leaf

    @property
    def grad_fn(self):
        return self._grad_fn


    @grad_fn.setter
    def grad_fn(self, new_value):
        self._grad_fn = new_value


    @property
    def T(self):
        return Tensor(
            data=self.data.T,
            requires_grad=self.requires_grad,
            grad_fn=self.grad_fn,
            is_leaf=self.is_leaf
        )


    @staticmethod
    def zeros_like(other: Self, requires_grad: bool = False):
        return Tensor(
            data=np.zeros_like(other.data), 
            requires_grad=requires_grad
        )

    
    @staticmethod
    def ones_like(other: Self, requires_grad: bool = False):
        return Tensor(
            data=np.ones_like(other.data), 
            requires_grad=requires_grad
        )


    @staticmethod
    def scalar_like(
        other: Self, 
        value: numbers.Number,
        requires_grad: bool = False
    ):
        return Tensor(
            data=np.ones_like(other.data) * value, 
            requires_grad=requires_grad
        )


    @staticmethod
    def tensor(
        other: Self,
        requires_grad: bool = True
    ):
        return Tensor(data=other.data, requires_grad=requires_grad)

    
    # Methods
    
    # Operands
    
    def __add__(self, other: Union[Self, numbers.Number]):
        if isinstance(other, (numbers.Number)):
            if self.requires_grad:
                Variable()
            