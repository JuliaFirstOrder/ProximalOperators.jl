# L-infinity norm

export NormLinf

"""
**``L_∞`` norm**

    NormLinf(λ=1.0)

Returns the function
```math
f(x) = λ⋅\\max\\{|x_1|, …, |x_n|\\},
```
for a nonnegative parameter `λ`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = Conjugate(IndBallL1(lambda))

function (f::Conjugate{IndBallL1{R}}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S})
  return (f.f.r)*vecnorm(x, Inf)
end

function gradient!{T <: RealOrComplex, R <: Real}(y::AbstractArray{T}, f::Conjugate{IndBallL1{R}}, x::AbstractArray{T})
  absxi, i = findmax(abs(xi) for xi in x) # Largest absolute value
  y .= 0
  y[i] = f.f.r*sign(x[i])
  return f.f.r*absxi
end

fun_name{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "weighted L-infinity norm"
fun_expr{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "λ = $(f.a*(f.f.f.r))"
