# Max function

export Maximum

"""
    Maximum(λ=1)

For a nonnegative parameter `λ ⩾ 0`, return the function
```math
f(x) = \\lambda \\cdot \\max \\{x_i : i = 1,\\ldots, n \\}.
```
"""
Maximum(lambda=1) = SumLargest(one(Int32), lambda)
