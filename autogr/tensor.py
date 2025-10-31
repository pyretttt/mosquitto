from typing import List
import numpy as np
import numbers

class Tensor():
    def __init__(
        self,
        data: np.array,
        requires_grad: bool
    ):
        self.data = data
        self.requires_grad = requires_grad
        self._grad_fn = None


    @property
    def requires_grad(self):
        return self._requires_grad


    @requires_grad.setter
    def requires_grad(self, new_value: bool):
        self._requires_grad = new_value
        if new_value:
            self.gradients = np.zeros_like(self.data)
        else:
            self.gradients = None
            self.grad_fn = None


    @property
    def grad_fn(self):
        return self._grad_fn


    @grad_fn.setter
    def grad_fn(self, new_value):
        self._grad_fn = new_value


    @staticmethod
    def zeros_like(other, requires_grad: bool = False):
        return Tensor(data=np.zeros_like(other.data), requires_grad=requires_grad)

    
    @staticmethod
    def ones_like(other, requires_grad: bool = False):
        return Tensor(data=np.ones_like(other.data), requires_grad=requires_grad)


    @staticmethod
    def scalar_like(other, value: numbers.Number, requires_grad: bool = False):
        return Tensor(data=np.ones_like(other.data) * value, requires_grad=requires_grad)


    @staticmethod
    def tesnor(
        other, 
        requires_grad: bool = True
    ):
        return Tensor(data=np.zeros_like(other.data), requires_grad=requires_grad)

    
    # Operands
    
    def __add__(self, other):
        if isinstance(other, (numbers.Integral, numbers.Rational)):
            op1 = 
            op1 = Tensor.scalar_like(self.data, value=other)