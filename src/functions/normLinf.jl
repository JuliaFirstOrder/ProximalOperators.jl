# L-infinity norm

"""
  NormLinf(λ::Real=1.0)

Returns the function `g(x) = λ⋅max(abs(x))`, for a nonnegative parameter `λ ⩾ 0`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = Postcomposition(Conjugate(IndBallL1(one(R))), lambda)

@compat function (f::Conjugate{IndBallL1{R}}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S})
  # follows from the definition of conjugate function and properties of norms
  return vecnorm(x, Inf)/(f.f.r)
end

fun_name{R <: Real}(f::Postcomposition{Conjugate{IndBallL1{R}}, R}) = "weighted L-infinity norm"
fun_expr{R <: Real}(f::Postcomposition{Conjugate{IndBallL1{R}}, R}) = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params{R <: Real}(f::Postcomposition{Conjugate{IndBallL1{R}}, R}) = "λ = $(f.a)"

function prox_naive{T <: RealOrComplex}(f::Postcomposition{Conjugate{IndBallL1}}, x::AbstractArray{T,1}, gamma::Real=1.0)
  return prox_naive(f.g, x, gamma)
end
