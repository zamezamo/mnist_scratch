from layer import Layer

class Optimizer:
    def step(self, layer: Layer) -> None:
        raise NotImplementedError