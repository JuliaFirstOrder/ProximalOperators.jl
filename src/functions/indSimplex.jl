# indicator of the probability simplex

"""
  IndSimplex(a::Real=1.0)

Returns the function `g = ind{x : x ⩾ 0, sum(x) = a}`.
"""

immutable IndSimplex <: IndicatorConvex
  a::Real
  function IndSimplex(a::Real=1.0)
    if a <= 0
      error("parameter a must be positive")
    else
      new(a)
    end
  end
end

@compat function (f::IndSimplex){T <: Real}(x::AbstractArray{T,1})
  if all(x .>= 0) && abs(sum(x)-f.a) <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: Real}(f::IndSimplex, x::AbstractArray{T,1}, y::AbstractArray{T}, gamma::Real=1.0)
# Implements Algorithm 1 in Condat, "Fast projection onto the simplex and the l1 ball", Mathematical Programming, 158:575–585, 2016.
# We should consider implementing the other algorithms reviewed there, and the one proposed in the paper.
  n = length(x)
  p = sort(x, rev=true);
  s = 0
  for i = 1:n-1
    s = s + p[i]
    tmax = (s - f.a)/i
    if tmax >= p[i+1]
      @inbounds for j in eachindex(y)
        y[j] = x[j] < tmax ? 0.0 : x[j] - tmax
      end
      return 0.0
    end
  end
  tmax = (s + p[n] - f.a)/n
  @inbounds for j in eachindex(y)
    y[j] = x[j] < tmax ? 0.0 : x[j] - tmax
  end
  return 0.0
end

fun_name(f::IndSimplex) = "indicator of the probability simplex"
fun_type(f::IndSimplex) = "Array{Real,1} → Real ∪ {+∞}"
fun_expr(f::IndSimplex) = "x ↦ 0 if x ⩾ 0 and sum(x) = 1, +∞ otherwise"
fun_params(f::IndSimplex) = "none"

function prox_naive{T <: Real}(f::IndSimplex, x::AbstractArray{T,1}, gamma::Real=1.0)
# Use bisection algorithm here? Like in IndBallL1
  n = length(x)
  p = sort(x,rev=true);
  s = 0
  for i = 1:n-1
    tmax = (sum(p[1:i]) - f.a)/i
    if tmax >= p[i+1]
      return (max(x-tmax,0), 0.0)
    end
  end
  tmax = (sum(p) - f.a)/n
  return max(x-tmax,0), 0.0
end
