from tensor import Tensor
from tensor_interactions import tensor_combine, tensor_sum
from loss import Loss

class SSE(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual
        )
        return tensor_sum(squared_errors)
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual
        )