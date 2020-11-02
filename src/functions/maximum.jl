# Max function

export Maximum

"""
**Maximum coefficient**

    Maximum(λ=1)

For a nonnegative parameter `λ ⩾ 0`, returns the function
```math
f(x) = \\lambda \\cdot \\max \\{x_i : i = 1,\\ldots, n \\}.
```
"""
Maximum(lambda::R=1) where {R <: Real} = SumLargest(one(Int32), lambda)
