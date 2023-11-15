from typing import Iterable
from tensor import Tensor
from layer import Layer
from tensor_interactions import random_tensor
from vector_interactions import dot

class Linear(Layer):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 init: str = 'xavier') -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = random_tensor(output_dim, input_dim, init=init)

        self.bias = random_tensor(output_dim, init=init)
    
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return [dot(input, self.weights[o]) + self.bias[o] for o in range(self.output_dim)]
    
    def backward(self, gradient: Tensor) -> Tensor:
        self.b_grad = gradient
        self.w_grad = [[self.input[i] * gradient[o] for i in range(self.input_dim)] for o in range(self.output_dim)]
        return [sum(self.weights[o][i] * gradient[o] for o in range(self.output_dim)) for i in range(self.input_dim)]
    
    def params(self) -> Iterable[Tensor]:
        return [self.weights, self.bias]
    
    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]