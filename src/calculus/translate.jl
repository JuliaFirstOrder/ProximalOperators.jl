export Translate

"""
**Translation**

    Translate(f, p=0.0)

Returns the translated function
```math
g(x) = f(x + b)
```
"""

Translate(f::ProximableFunction, p=0.0) = PrecomposeDiagonal(f, 1.0, p)
