# Hinge loss function

export HingeLoss

"""
**Hinge loss**

	HingeLoss(y, μ=1.0)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\},
```
where `y` is an array and `μ` is a positive parameter.
"""
HingeLoss(y::T, mu::R=1.0) where {T <: AbstractArray, R <: Real} = Postcompose(PrecomposeDiagonal(SumPositive(), -y, 1.0), mu)
