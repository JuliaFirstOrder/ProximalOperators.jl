# L2 norm (times a constant)

"""
  NormL2(λ::Real=1.0)

Returns the function `g(x) = λ||x||_2`, for a real parameter `λ ⩾ 0`.
"""

immutable NormL2{R <: Real} <: ProximableConvex
  lambda::R
  function NormL2(lambda::R)
    if lambda < 0
      error("parameter λ must be nonnegative")
    else
      new(lambda)
    end
  end
end

NormL2{R <: Real}(lambda::R=1.0) = NormL2{R}(lambda)

function (f::NormL2)(x::AbstractArray)
  return f.lambda*vecnorm(x)
end

function prox!{T <: RealOrComplex}(f::NormL2, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  vecnormx = vecnorm(x)
  scale = max(0, 1-f.lambda*gamma/vecnormx)
  for i in eachindex(x)
    y[i] = scale*x[i]
  end
  return f.lambda*scale*vecnormx
end

fun_name(f::NormL2) = "Euclidean norm"
fun_dom(f::NormL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::NormL2) = "x ↦ λ||x||_2"
fun_params(f::NormL2) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::NormL2, x::AbstractArray{T}, gamma::Real=1.0)
  vecnormx = vecnorm(x)
  scale = max(0, 1-f.lambda*gamma/vecnormx)
  y = scale*x
  return y, f.lambda*scale*vecnormx
end
