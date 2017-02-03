# Hinge loss function

"""
  HingeLoss(b::Array{Real}, mu::Real=1.0)

Returns the function `g(x) = mu * sum(max(0, 1 - b_i * x_i), i=1,...,n )`.
"""

HingeLoss{T <: AbstractArray, R <: Real}(b::T, mu::R=1.0) = Postcomposition(Precomposition(SumPositive(), -b, 1.0), mu)
