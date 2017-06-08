# L-infinity norm

export NormLinf

"""
  NormLinf(λ::Real=1.0)

Returns the function `g(x) = λ⋅max(abs(x))`, for a nonnegative parameter `λ ⩾ 0`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = Conjugate(IndBallL1(lambda))

function (f::Conjugate{IndBallL1{R}}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S})
  return (f.f.r)*vecnorm(x, Inf)
end

fun_name{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "weighted L-infinity norm"
fun_expr{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "λ = $(f.a*(f.f.f.r))"
