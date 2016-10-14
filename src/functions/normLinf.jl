# L-infinity norm

immutable NormLinf{R <: Real} <: ProximableFunction
  g::ProximableFunction
  lambda::R
end

"""
  NormLinf(λ::Real=1.0)

Returns the function `g(x) = λ*max(abs(x))`, for a nonnegative parameter `λ ⩾ 0`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = NormLinf{R}(Postcomposition(Conjugate(IndBallL1(one(R))), lambda), lambda)

@compat function (f::NormLinf{R}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S,1})
  return f.lambda*maximum(abs(x))
end

prox!{T <: RealOrComplex}(f::NormLinf, x::AbstractArray{T,1}, y::AbstractArray{T,1}, gamma::Real=1.0) =
  prox!(f.g, x, y, gamma)

fun_name(f::NormLinf) = "weighted L-infinity norm"
fun_type(f::NormLinf) = "Array{Complex} → Real"
fun_expr(f::NormLinf) = "x ↦ λ||x||_∞"
fun_params(f::NormLinf) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::NormLinf, x::AbstractArray{T,1}, gamma::Real=1.0)
  return prox_naive(f.g, x, gamma)
end
