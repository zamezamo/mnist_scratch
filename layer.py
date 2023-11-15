from typing import Iterable, Tuple

from tensor import Tensor

class Layer:

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradient):
        raise NotImplementedError
    
    def params(self) -> Iterable[Tensor]:
        return()
    
    def grads(self) -> Iterable[Tensor]:
        return()