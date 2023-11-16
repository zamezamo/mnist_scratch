from sigmoid import sigmoid
from tensor import Tensor
from tensor_interactions import tensor_apply, tensor_combine
from layer import Layer

class Sigmoid(Layer):
    
    def forward(self, input: Tensor) -> Tensor:
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids
    
    def backward(self, gradient: Tensor):
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids, gradient)