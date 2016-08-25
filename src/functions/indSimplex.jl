# indicator of the probability simplex

"""
  IndSimplex()

Returns the function `g = ind{x : x ⩾ 0, sum(x) = 1}`.
"""

immutable IndSimplex <: IndicatorConvex end

function call(f::IndSimplex, x::Array{Float64,1})
  if all(x .>= 0) && abs(sum(x)-1) <= 1e-14 return 0.0 end
  return +Inf
end

function prox(f::IndSimplex, x::Array{Float64}, gamma::Float64=1.0)
  n = length(x)
  p = sort(x,rev=true);
  s = 0
  for i = 1:n-1
    s = s + p[i]
    tmax = (s - 1)/i
    if tmax >= p[i+1] return (max(x-tmax,0), 0.0) end
  end
  tmax = (s + p(n) - 1)/n
  return max(x-tmax,0), 0.0
end

fun_name(f::IndSimplex) = "indicator of the probability simplex"
fun_type(f::IndSimplex) = "R^n → R ∪ {+∞}"
fun_expr(f::IndSimplex) = "x ↦ 0 if x ⩾ 0 and sum(x) = 1, +∞ otherwise"
fun_params(f::IndSimplex) = "none"
