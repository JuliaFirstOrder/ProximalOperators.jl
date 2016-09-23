# indicator of the L2 norm sphere with given radius

"""
  IndSphereL2(r::Real=1.0)

Returns the function `g = ind{x : ||x|| = r}`, for a real parameter `r > 0`.
"""

immutable IndSphereL2 <: IndicatorFunction
  r::Real
  IndSphereL2(r::Real=1.0) =
    r <= 0 ? error("parameter r must be positive") : new(r)
end

@compat function (f::IndSphereL2)(x::RealOrComplexArray)
  if abs(vecnorm(x) - f.r)/f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplexArray}(f::IndSphereL2, x::T, gamma::Real, y::T)
  normx = vecnorm(x)
  if normx > 0
    scal = f.r/normx
    for k in eachindex(x)
      y[k] = scal*x[k]
    end
  else
    # TODO: pick y as a uniform random point on the sphere
  end
  return 0.0
end

fun_name(f::IndSphereL2) = "indicator of an L2 norm sphere"
fun_type(f::IndSphereL2) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(f::IndSphereL2) = "x ↦ 0 if ||x|| = r, +∞ otherwise"
fun_params(f::IndSphereL2) = "r = $(f.r)"

function prox_naive(f::IndSphereL2, x::RealOrComplexArray, gamma::Real=1.0)
  normx = vecnorm(x)
  if normx > 0
    y = x*f.r/normx
  else
    # TODO: pick y as a uniform random point on the sphere
  end
  return y, 0.0
end
