from tanh import tanh
from tensor import Tensor
from tensor_interactions import tensor_apply, tensor_combine
from layer import Layer

class Tanh(Layer):
    
    def forward(self, input: Tensor) -> Tensor:
        self.tanhs = tensor_apply(tanh, input)
        return self.tanhs
    
    def backward(self, gradient: Tensor):
        return tensor_combine(lambda tanh, grad: (1 - tanh ** 2) * grad,
                              self.tanhs, gradient)