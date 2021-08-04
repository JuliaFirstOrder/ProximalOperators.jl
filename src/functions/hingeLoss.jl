# Hinge loss function

export HingeLoss

"""
**Hinge loss**

    HingeLoss(y, μ=1)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\},
```
where `y` is an array and `μ` is a positive parameter.
"""
HingeLoss(y) = PrecomposeDiagonal(SumPositive(), -y, 1)

HingeLoss(y, mu) = Postcompose(PrecomposeDiagonal(SumPositive(), -y, 1), mu)
