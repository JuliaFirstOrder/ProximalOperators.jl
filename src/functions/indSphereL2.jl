# indicator of the L2 norm sphere with given radius

export IndSphereL2

"""
**Indicator of a Euclidean sphere**

    IndSphereL2(r=1.0)

Returns the indicator function of the set
```math
S = \\{ x : \\|x\\| = r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.
"""

struct IndSphereL2{R <: Real} <: ProximableFunction
  r::R
  function IndSphereL2{R}(r::R) where {R <: Real}
    if r <= 0
      error("parameter r must be positive")
    else
      new(r)
    end
  end
end

is_set(f::IndSphereL2) = true

IndSphereL2(r::R=1.0) where {R <: Real} = IndSphereL2{R}(r)

function (f::IndSphereL2)(x::AbstractArray{T}) where T <: RealOrComplex
  if abs(vecnorm(x) - f.r)/f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!(y::AbstractArray{T}, f::IndSphereL2, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
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
fun_dom(f::IndSphereL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndSphereL2) = "x ↦ 0 if ||x|| = r, +∞ otherwise"
fun_params(f::IndSphereL2) = "r = $(f.r)"

function prox_naive(f::IndSphereL2, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
  normx = vecnorm(x)
  if normx > 0
    y = x*f.r/normx
  else
    y = randn(size(x))
    y *= f.r/vecnorm(y)
  end
  return y, 0.0
end
