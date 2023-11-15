from typing import List, Callable

from tensor import Tensor
from probability import inverse_normal_cdf

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

assert shape([[1, 2], [3, 4]]) == [2, 2]

def is_onedim(tensor: Tensor) -> bool:
    return not isinstance(tensor[0], list)

def tensor_sum(tensor: Tensor) -> float: 
    if is_onedim(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)
    
assert tensor_sum([[1, 2], [3, 4]]) == 10

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    if is_onedim(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]
    
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def tensor_combine(f: Callable[[float, float], float], t1: Tensor, t2: Tensor) -> Tensor:
    if is_onedim(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

import operator    
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]
    
def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

import random

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]

def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])]

assert shape(random_normal(2, 3, 4, mean=10)) == [2, 3, 4]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"Unknown init: {init}")