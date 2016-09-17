# indicator of the L0 norm ball with given (integer) radius

"""
  IndBallL0(r::Int=1)

Returns the function `g = ind{x : countnz(x) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallL0 <: IndicatorFunction
  r::Int
  IndBallL0(r::Int=1) =
    r <= 0 ? error("parameter r must be a positive integer") : new(r)
end

@compat function (f::IndBallL0)(x::RealOrComplexArray)
  if countnz(x) > f.r return +Inf end
  return 0.0
end

function prox!(f::IndBallL0, x::RealOrComplexArray, gamma::Float64, y::RealOrComplexArray)
  p = sortperm(abs(x)[:], rev=true)
  y[p[1:f.r]] = x[p[1:f.r]]
  y[p[f.r+1:end]] = 0
  return 0.0
end

fun_name(f::IndBallL0) = "indicator of an L0 pseudo-norm ball"
fun_type(f::IndBallL0) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(f::IndBallL0) = "x ↦ 0 if countnz(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallL0) = "r = $(f.r)"
