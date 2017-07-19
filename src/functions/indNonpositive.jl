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

immutable IndNonpositive <: ProximableFunction end

is_separable(f::IndNonpositive) = true
is_convex(f::IndNonpositive) = true
is_cone(f::IndNonpositive) = true

function (f::IndNonpositive){R <: Real}(x::AbstractArray{R})
  for k in eachindex(x)
    if x[k] > 0
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: Real}(y::AbstractArray{R}, f::IndNonpositive, x::AbstractArray{R}, gamma::Real=1.0)
  for k in eachindex(x)
    if x[k] > 0
      y[k] = zero(R)
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

prox!{R <: Real}(y::AbstractArray{R}, f::IndNonpositive, x::AbstractArray{R}, gamma::AbstractArray) = prox!(y, f, x, 1.0)

fun_name(f::IndNonpositive) = "indicator of the Nonpositive cone"
fun_dom(f::IndNonpositive) = "AbstractArray{Real}"
fun_expr(f::IndNonpositive) = "x ↦ 0 if all(0 ⩾ x), +∞ otherwise"
fun_params(f::IndNonpositive) = "none"

function prox_naive{R <: Real}(f::IndNonpositive, x::AbstractArray{R}, gamma::Real=1.0)
  y = min.(zero(R), x)
  return y, 0.0
end

prox_naive{R <: Real}(f::IndNonpositive, x::AbstractArray{R}, gamma::AbstractArray) = prox_naive(f, x, 1.0)
