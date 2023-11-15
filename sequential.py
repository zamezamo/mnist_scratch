from typing import Iterable, List
from tensor import Tensor

from layer import Layer

class Sequential(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.bacward(gradient)
        return gradient
    
    def params(self) -> Iterable[Tensor]:
        return (param for layer in self.layers for param in layer.params())
    
    def grads(self) -> Iterable[Tensor]:
        return (grad for layer in self.layers for grad in layer.grads())