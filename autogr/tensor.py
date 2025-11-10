from typing import Optional, Self, Union, List
import numpy as np
import numbers

from grad import Variable
import methods

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
        self.grad: Self = None

    @property
    def grad_fn(self):
        return self._grad_fn


    @grad_fn.setter
    def grad_fn(self, new_value):
        self._grad_fn = new_value


    @property
    def T(self) -> Self:
        return Tensor(
            data=self.data.T,
            requires_grad=self.requires_grad,
            grad_fn=self.grad_fn,
            is_leaf=self.is_leaf
        )

    @property
    def shape(self):
        return self.data.shape

    # Constructors

    @staticmethod
    def zeros_like(
        other: Self,
        requires_grad: bool = False,
        is_leaf: bool = True
    ) -> Self:
        return Tensor(
            data=np.zeros_like(other.data),
            requires_grad=requires_grad,
            is_leaf=is_leaf
        )


    @staticmethod
    def ones_like(
        other: Self,
        requires_grad: bool = False,
        is_leaf: bool = True
    ) -> Self:
        return Tensor(
            data=np.ones_like(other.data),
            requires_grad=requires_grad,
            is_leaf=is_leaf
        )


    @staticmethod
    def scalar_like(
        other: Self,
        value: numbers.Number,
        requires_grad: bool = False,
        is_leaf: bool = True
    ) -> Self:
        return Tensor(
            data=np.ones_like(other.data) * value,
            requires_grad=requires_grad,
            is_leaf=is_leaf
        )


    @staticmethod
    def tensor(
        other: Self,
        requires_grad: bool = False,
        is_leaf: bool = True
    ) -> Self:
        return Tensor(data=other.data, requires_grad=requires_grad, is_leaf=is_leaf)


    @staticmethod
    def from_numpy(
        data: np.array,
        requires_grad: bool = False,
        is_leaf: bool = True
    ) -> Self:
        return Tensor(
            data=data,
            requires_grad=requires_grad,
            is_leaf=is_leaf
        )


    @staticmethod
    def diag(
        data: Union[np.array, List[numbers.Number], numbers.Number, Self],
        requires_grad: bool = False,
        is_leaf: bool = True
    ):
        return Tensor(
            data=np.diag(data),
            requires_grad=requires_grad,
            is_leaf=is_leaf
        )

    # Methods

    def backward(self):
        if self.grad_fn is None:
            raise RuntimeError("backward is called when gradient is not computed")
        self.grad_fn.backward(chain_jacobian=None)


    # Operands

    def __add__(self, other: Union[Self, numbers.Number]):
        return methods.add(self, other)


    # Meta

    def __repr__(self):
        return str(
            dict(
                data=self.data,
                requires_grad=self.requires_grad,
                is_leaf=self.is_leaf,
                grad_fn=self.grad_fn
            )
        )