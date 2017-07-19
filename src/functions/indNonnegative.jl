# indicator of nonnegative orthant

export IndNonnegative

"""
**Indicator of the nonnegative orthant**

    IndNonnegative()

Returns the indicator of the set
```math
C = \\{ x : x \\geq 0 \\}.
```
"""

immutable IndNonnegative <: ProximableFunction end

is_separable(f::IndNonnegative) = true
is_convex(f::IndNonnegative) = true
is_cone(f::IndNonnegative) = true

function (f::IndNonnegative){R <: Real}(x::AbstractArray{R})
  for k in eachindex(x)
    if x[k] < 0
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: Real}(y::AbstractArray{R}, f::IndNonnegative, x::AbstractArray{R}, gamma::Real=1.0)
  for k in eachindex(x)
    if x[k] < 0
      y[k] = zero(R)
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

prox!{R <: Real}(y::AbstractArray{R}, f::IndNonnegative, x::AbstractArray{R}, gamma::AbstractArray{R}) = prox!(y, f, x, 1.0)

fun_name(f::IndNonnegative) = "indicator of the Nonnegative cone"
fun_dom(f::IndNonnegative) = "AbstractArray{Real}"
fun_expr(f::IndNonnegative) = "x ↦ 0 if all(0 ⩽ x), +∞ otherwise"
fun_params(f::IndNonnegative) = "none"

function prox_naive{R <: Real}(f::IndNonnegative, x::AbstractArray{R}, gamma::Real=1.0)
  y = max.(zero(R), x)
  return y, 0.0
end

prox_naive{R <: Real}(f::IndNonnegative, x::AbstractArray{R}, gamma::AbstractArray) = prox_naive(f, x, 1.0)
