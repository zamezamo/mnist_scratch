from vector import Vector

def dot(v1: Vector, v2: Vector) -> float:
    assert len(v1) == len(v2)
    return sum(v1_i * v2_i for v1_i, v2_i in zip(v1, v2))

assert dot([1, 2, 3], [4, 5, 6]) == 32