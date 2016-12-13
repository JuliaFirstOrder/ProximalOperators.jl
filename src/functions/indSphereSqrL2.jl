
# indicator of the squared L2 norm sphere with given radius

"""
  IndSphereSqrL2(r::Real=1.0)

  Returns the function `g = ind{x : dot(x,x) = r}`, for a real parameter `r > 0`.
"""

immutable IndSphereSqrL2{R <: Real} <: IndicatorFunction
  r::R
  function IndSphereSqrL2(r::R)
    if r <= 0
      error("parameter r must be positive")
    else
      new(r)
    end
  end
end

IndSphereSqrL2{R <: Real}(r::R=1.0) = IndSphereSqrL2{R}(r)

function (f::IndSphereSqrL2){T <: RealOrComplex}(x::AbstractArray{T})
  if abs(vecnorm(x)^2 - f.r)/f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(f::IndSphereSqrL2, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  normx = vecdot(x,x)
  if normx > 0 # zero-zero?
    scal = sqrt(f.r/normx)
    for k in eachindex(x)
      y[k] = scal*x[k]
    end
  else
    normy = 0.0
    for k in eachindex(x)
      y[k] = randn()
      normy += y[k]*y[k]
    end
    y[:] *= sqrt(f.r/normy)
  end
  return 0.0
end

fun_name(f::IndSphereSqrL2) = "indicator of a squared L2 norm sphere"
fun_dom(f::IndSphereSqrL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndSphereSqrL2) = "x ↦ 0 if dot(x,x) = r, +∞ otherwise"
fun_params(f::IndSphereSqrL2) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndSphereSqrL2, x::AbstractArray{T}, gamma::Real=1.0)
  normx = vecdot(x,x)
  if normx > 0
	  y = x*sqrt(f.r/normx)
  else
    y = randn(size(x))
    y *= sqrt(f.r/dot(y,y))
  end
  return y, 0.0
end
