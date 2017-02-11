# L-infinity norm

"""
  NormLinf(λ::Real=1.0)

Returns the function `g(x) = λ⋅max(abs(x))`, for a nonnegative parameter `λ ⩾ 0`.
"""

NormLinf{R <: Real}(lambda::R=1.0) = Postcompose(Conjugate(IndBallL1(one(R))), lambda)

function (f::Conjugate{IndBallL1{R}}){R <: Real, S <: RealOrComplex}(x::AbstractArray{S})
  # follows from the definition of conjugate function and properties of norms
  # although it is not really needed since with the above constructor one makes f.f.r = 1.0
  # but just in case one constructs the conjugate of a different L1 ball...
  return vecnorm(x, Inf)*(f.f.r)
end

fun_name{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "weighted L-infinity norm"
fun_expr{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params{R <: Real}(f::Postcompose{Conjugate{IndBallL1{R}}, R}) = "λ = $(f.a*(f.f.f.r))"
