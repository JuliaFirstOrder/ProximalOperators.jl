# indicator of nonpositive orthant

export IndNonpositive

"""
**Indicator of the nonpositive orthant**

    IndNonpositive()

Returns the indicator of the set
```math
C = \\{ x : x \\leq 0 \\}.
```
"""

struct IndNonpositive <: ProximableFunction end

is_separable(f::IndNonpositive) = true
is_convex(f::IndNonpositive) = true
is_cone(f::IndNonpositive) = true

function (f::IndNonpositive)(x::AbstractArray{R}) where R <: Real
  for k in eachindex(x)
    if x[k] > 0
      return +Inf
    end
  end
  return zero(R)
end

function prox!(y::AbstractArray{R}, f::IndNonpositive, x::AbstractArray{R}, gamma::Real=1.0) where R <: Real
  for k in eachindex(x)
    if x[k] > 0
      y[k] = zero(R)
    else
      y[k] = x[k]
    end
  end
  return zero(R)
end

prox!(y::AbstractArray{R}, f::IndNonpositive, x::AbstractArray{R}, gamma::AbstractArray) where {R <: Real} = prox!(y, f, x, 1.0)

fun_name(f::IndNonpositive) = "indicator of the Nonpositive cone"
fun_dom(f::IndNonpositive) = "AbstractArray{Real}"
fun_expr(f::IndNonpositive) = "x ↦ 0 if all(0 ⩾ x), +∞ otherwise"
fun_params(f::IndNonpositive) = "none"

function prox_naive(f::IndNonpositive, x::AbstractArray{R}, gamma::Real=1.0) where R <: Real
  y = min.(zero(R), x)
  return y, 0.0
end

prox_naive(f::IndNonpositive, x::AbstractArray{R}, gamma::AbstractArray) where {R <: Real} = prox_naive(f, x, 1.0)
