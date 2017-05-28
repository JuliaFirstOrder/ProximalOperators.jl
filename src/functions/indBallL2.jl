# indicator of the L2 norm ball with given radius

"""
  IndBallL2(r::Real=1.0)

Returns the function `g = ind{x : ||x|| ⩽ r}`, for a real parameter `r > 0`.
"""

immutable IndBallL2{R <: Real} <: ProximableFunction
  r::R
  function IndBallL2{R}(r::R) where {R <: Real}
    if r <= 0
      error("parameter r must be positive")
    else
      new(r)
    end
  end
end

is_convex(f::IndBallL2) = true
is_set(f::IndBallL2) = true

IndBallL2{R <: Real}(r::R=1.0) = IndBallL2{R}(r)

function (f::IndBallL2){T <: RealOrComplex}(x::AbstractArray{T})
  if vecnorm(x) - f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, f::IndBallL2, x::AbstractArray{T}, gamma::Real=1.0)
  scal = f.r/vecnorm(x)
  if scal > 1
    y[:] = x
    return 0.0
  end
  for k in eachindex(x)
    y[k] = scal*x[k]
  end
  return 0.0
end

fun_name(f::IndBallL2) = "indicator of an L2 norm ball"
fun_dom(f::IndBallL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndBallL2) = "x ↦ 0 if ||x|| ⩽ r, +∞ otherwise"
fun_params(f::IndBallL2) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndBallL2, x::AbstractArray{T}, gamma::Real=1.0)
  normx = vecnorm(x)
  if normx > f.r
    y = (f.r/normx)*x
  else
    y = x
  end
  return y, 0.0
end
