from tensor import Tensor

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError