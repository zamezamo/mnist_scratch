import math
def tanh(x: float) -> float:
    if x < -100: return -1
    elif x > 100: return 1
    
    ex = math.exp(-2 * x)
    
    return (1 - ex) / (1 + ex)