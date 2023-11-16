from tensor import Tensor
from tensor_interactions import tensor_apply, tensor_combine
from layer import Layer
import random
import operator

class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True
        
    def forward(self, input: Tensor):
        if self.train:
            self.mask = tensor_apply(lambda _: 0 if random.random() < self.p else 1, input)
            return tensor_combine(operator.mul, input, self.mask)
        else:
            return tensor_combine(lambda x: x * (1 - self.p), input)
        
    def backward(self, gradient):
        if self.train:
            return tensor_combine(operator.mul, gradient, self.mask)
        else:
            raise RuntimeError("don't call backward in train mode")