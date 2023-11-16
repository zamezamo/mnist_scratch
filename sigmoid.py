from math import exp

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))