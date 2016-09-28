# indicator of the L2 norm sphere with given radius

"""
  IndSphereL2(r::Real=1.0)

Returns the function `g = ind{x : ||x|| = r}`, for a real parameter `r > 0`.
"""

immutable IndSphereL2 <: IndicatorFunction
  r::Real
  function IndSphereL2(r::Real=1.0)
    if r <= 0
      error("parameter r must be positive")
    else
      new(r)
    end
  end
end

@compat function (f::IndSphereL2){T <: RealOrComplex}(x::AbstractArray{T})
  if abs(vecnorm(x) - f.r)/f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(f::IndSphereL2, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  normx = vecnorm(x)
  if normx > 0 # zero-zero?
    scal = f.r/normx
    for k in eachindex(x)
      y[k] = scal*x[k]
    end
  else
    normy = 0.0
    for k in eachindex(x)
      y[k] = randn()
      normy += y[k]*y[k]
    end
    normy = sqrt(normy)
    y[:] *= f.r/normy
  end
  return 0.0
end

fun_name(f::IndSphereL2) = "indicator of an L2 norm sphere"
fun_type(f::IndSphereL2) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(f::IndSphereL2) = "x ↦ 0 if ||x|| = r, +∞ otherwise"
fun_params(f::IndSphereL2) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndSphereL2, x::AbstractArray{T}, gamma::Real=1.0)
  normx = vecnorm(x)
  if normx > 0
    y = x*f.r/normx
  else
    y = randn(size(x))
    y *= f.r/vecnorm(y)
  end
  return y, 0.0
end
