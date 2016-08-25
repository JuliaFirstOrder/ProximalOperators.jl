# indicator of second-order cones

"""
  IndSOC(n::Int64)

Returns the indicator of the second-order cone (ice-cream cone) of R^n.
"""

immutable IndSOC <: IndicatorConvex
  n::Int64
  IndSOC(n::Int64) =
    new(n)
end

function call(f::IndSOC, x::Array{Float64,1})
  # the tolerance in the following line should be customizable
  if norm(x[2:end]) - x[1] <= 1e-14 return 0.0 end
  return +Inf
end

function prox(f::IndSOC, x::Array{Float64,1}, gamma::Float64=1.0)
  nx = norm(x[2:end])
  t = x[1]
  if nx <= -t
    y = zeros(x)
  elseif nx <= t
    y = x
  else
    y = zeros(x)
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    y[2:end] = r * x[2:end]
  end
  return y, 0.0
end

fun_name(f::IndSOC) = "indicator of a second-order cone"
fun_type(f::IndSOC) = "R^n → R ∪ {+∞}"
fun_expr(f::IndSOC) = "x ↦ 0 if x[1] >= ||x[2:end]||, +∞ otherwise"
fun_params(f::IndSOC) = "n = $(f.n)"
