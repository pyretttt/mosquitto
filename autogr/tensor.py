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
    
    
    @staticmethod
    def zeros_like(other, requires_grad: bool = False):
        return Tensor(data=np.zeros_like(other.data), requires_grad=requires_grad)
    
    
    @staticmethod
    def ones_like(other, requires_grad: bool = False):
        return Tensor(data=np.ones_like(other.data), requires_grad=requires_grad)

    @staticmethod
    def scalar_like(other, value: numbers.Number, requires_grad: bool = False):
        return Tensor(data=np.ones_like(other.data) * value, requires_grad=requires_grad)


    # Operands
    
    def __add__(self, other):
        if isinstance(other, (numbers.Integral, numbers.Rational)):
            constant = Tensor.scalar_like(self.data, value=other)
            argument = self
            