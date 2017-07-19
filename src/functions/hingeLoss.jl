# Hinge loss function

export HingeLoss

"""
**Hinge loss**

  HingeLoss(b, μ=1.0)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - b_i ⋅ x_i\\},
```
where `b` is an array and `μ` is a positive parameter.
"""

HingeLoss{T <: AbstractArray, R <: Real}(b::T, mu::R=1.0) = Postcompose(PrecomposeDiagonal(SumPositive(), -b, 1.0), mu)
