# L-infinity norm

immutable NormLinf{R <: Real}
  g::ProximableFunction
  lambda::R
end

"""
  NormLinf(λ::Real=1.0)

Returns the function `g(x) = λ*max(abs(x))`, for a nonnegative parameter `λ ⩾ 0`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = NormLinf{R}(Postcomposition(Conjugate(IndBallL1(one(R))), lambda), lambda)

@compat function (f::NormLinf{R}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S})
  return f.lambda*maximum(abs(x))
end

prox!{T <: RealOrComplex}(f::NormLinf, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0) =
  prox!(f.g, x, y, gamma)

fun_name(f::NormLinf) = "weighted L-infinity norm"
fun_type(f::NormLinf) = "Array{Complex} → Real"
fun_expr(f::NormLinf) = "x ↦ λ||x||_∞"
fun_params(f::NormLinf) = "λ = $(f.lambda)"
