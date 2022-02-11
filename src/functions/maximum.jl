# Max function

export Maximum

"""
    Maximum(λ=1)

For a nonnegative parameter `λ ⩾ 0`, return the function
```math
f(x) = \\lambda \\cdot \\max \\{x_i : i = 1,\\ldots, n \\}.
```
"""
Maximum(lambda::R=1) where R = SumLargest(one(Int32), lambda)
