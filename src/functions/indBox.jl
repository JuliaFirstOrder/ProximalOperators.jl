# indicator of a generic box

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox <: IndicatorConvex
  lb
  ub
  IndBox(lb,ub) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub)
end

function call(f::IndBox, x::Array{Float64})
  if any(x .< f.lb) || any(x .> f.ub) return +Inf end
  return 0.0
end

function prox(f::IndBox, gamma::Float64, x::Array{Float64})
  y = min(f.ub, max(f.lb, x))
  return y, 0.0
end

# indicator of the L-infinity ball (box centered in the origin)

"""
  IndBallInf(r::Float64)

Returns the indicator function of an infinity-norm ball, that is function
`g(x) = ind{maximum(abs(x)) ⩽ r}` for `r ⩾ 0`.
"""

IndBallInf(r::Float64) = IndBox(-r, r)

# indicator of the nonnegative orthant

"""
  IndNonnegative()

Returns the indicator function the nonnegative orthant, that is

  `g(x) = 0 if x ⩾ 0, +∞ otherwise`
"""

IndNonnegative() = IndBox(0, +Inf)

fun_name(f::IndBox) = "indicator of a box"
fun_type(f::IndBox) = "R^n → R ∪ {+∞}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
  string( "lb = ", typeof(f.lb) <: Array ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
          "ub = ", typeof(f.ub) <: Array ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)
