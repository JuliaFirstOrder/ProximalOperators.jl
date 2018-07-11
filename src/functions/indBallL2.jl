# indicator of the L2 norm ball with given radius

export IndBallL2

"""
**Indicator of a Euclidean ball**

    IndBallL2(r=1.0)

Returns the indicator function of the set
```math
S = \\{ x : \\|x\\| \\leq r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.
"""

struct IndBallL2{R <: Real} <: ProximableFunction
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

IndBallL2(r::R=1.0) where {R <: Real} = IndBallL2{R}(r)

function (f::IndBallL2)(x::AbstractArray{T}) where T <: RealOrComplex
  if vecnorm(x) - f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!(y::AbstractArray{T}, f::IndBallL2, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
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

function prox_naive(f::IndBallL2, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
  normx = vecnorm(x)
  if normx > f.r
    y = (f.r/normx)*x
  else
    y = x
  end
  return y, 0.0
end
