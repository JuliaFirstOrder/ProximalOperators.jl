# indicator of second-order cones

"""
  IndSOC()

Returns the indicator of the second-order cone (ice-cream cone) of R^n.
"""

immutable IndSOC <: IndicatorConvex end

@compat function (f::IndSOC)(x::Array{Float64,1})
  # the tolerance in the following line should be customizable
  if norm(x[2:end]) - x[1] <= 1e-14 return 0.0 end
  return +Inf
end

function prox!(f::IndSOC, x::Array{Float64,1}, gamma::Float64, y::Array{Float64,1})
  nx = norm(x[2:end])
  t = x[1]
  if t <= -nx
    y[:] = 0.0
  elseif t >= nx
    y[:] = x
  else
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    y[2:end] = r * x[2:end]
  end
  return 0.0
end

fun_name(f::IndSOC) = "indicator of a second-order cone"
fun_type(f::IndSOC) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndSOC) = "x ↦ 0 if x[1] >= ||x[2:end]||, +∞ otherwise"
fun_params(f::IndSOC) = "none"

function prox_naive(f::IndSOC, x::Array{Float64,1}, gamma::Float64=1.0)
  nx = norm(x[2:end])
  t = x[1]
  if t <= -nx
    y = zeros(x)
  elseif t >= nx
    y = x
  else
    y = zeros(x)
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    y[2:end] = r * x[2:end]
  end
  return y, 0.0
end
