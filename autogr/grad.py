from typing import Callable, Union, TypeVar
    

class Variable:
    def __init__(self, argument, do):
        self.do = do
        self.argument = argument
        
    def backward(self, **params):
        chain_jacobian = self.do(**params)
        if argument.