# indicator of a simplex

export IndSimplex

"""
**Indicator of a simplex**

    IndSimplex(a=1.0)

Returns the indicator of the set
```math
S = \\left\\{ x : x \\geq 0, ∑_i x_i = a \\right\\}.
```
By default `a=1.0`, therefore ``S`` is the probability simplex.
"""
struct IndSimplex{T <: Union{Real, Integer}} <: ProximableFunction
  a::T
  function IndSimplex{T}(a::T) where {T <: Union{Real, Integer}}
    if a <= 0
      error("parameter a must be positive")
    else
      new(a)
    end
  end
end

is_convex(f::IndSimplex) = true
is_set(f::IndSimplex) = true

IndSimplex(a::T=1.0) where {T <: Union{Real, Integer}} = IndSimplex{T}(a)

function (f::IndSimplex)(x::AbstractArray{T}) where T <: Real
  if all(x .>= 0) && abs(sum(x)-f.a) <= 1e-14
    return zero(T)
  end
  return +Inf
end

function prox!(y::AbstractArray{T}, f::IndSimplex, x::AbstractArray{T}, gamma::Real=1.0) where T <: Real
# Implements Algorithm 1 in Condat, "Fast projection onto the simplex and the l1 ball", Mathematical Programming, 158:575–585, 2016.
# We should consider implementing the other algorithms reviewed there, and the one proposed in the paper.
  n = length(x)
  p = []
  if ndims(x) == 1
    p = sort(x, rev=true)
  else
    p = sort(x[:], rev=true)
  end
  s = 0
  for i = 1:n-1
    s = s + p[i]
    tmax = (s - f.a)/i
    if tmax >= p[i+1]
      @inbounds for j in eachindex(y)
        y[j] = x[j] < tmax ? 0.0 : x[j] - tmax
      end
      return zero(T)
    end
  end
  tmax = (s + p[n] - f.a)/n
  @inbounds for j in eachindex(y)
    y[j] = x[j] < tmax ? 0.0 : x[j] - tmax
  end
  return zero(T)
end

fun_name(f::IndSimplex) = "indicator of the probability simplex"
fun_dom(f::IndSimplex) = "AbstractArray{Real}"
fun_expr(f::IndSimplex) = "x ↦ 0 if x ⩾ 0 and sum(x) = a, +∞ otherwise"
fun_params(f::IndSimplex) = "a = $(f.a)"

function prox_naive(f::IndSimplex, x::AbstractArray{T}, gamma::Real=1.0) where T <: Real
  low = minimum(x)
  upp = maximum(x)
  v = x
  s = Inf
  for i = 1:100
    if abs(s)/f.a <= 1e-15
      break
    end
    alpha = (low+upp)/2
    v = max.(x .- alpha, 0.0)
    s = sum(v) - f.a
    if s <= 0
      upp = alpha
    else
      low = alpha
    end
  end
  return v, 0.0
end
