from typing import List

from tensor import Tensor
from tensor_interactions import zeros_like, tensor_combine
from layer import Layer
from optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []
        
    def step(self, layer: Layer) -> None:
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad
            )
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update
            )