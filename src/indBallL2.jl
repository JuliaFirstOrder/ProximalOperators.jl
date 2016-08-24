# indicator of the L2 norm ball with given radius

"""
  IndBallL2(r::Float64)

Returns the function `g = ind{x : ||x|| ⩽ r}`, for a real parameter `r > 0`.
"""

immutable IndBallL2 <: IndicatorConvex
  r::Float64
  IndBallL2(r::Float64) =
    r <= 0 ? error("parameter r must be positive") : new(r)
end

function call(f::IndBallL2, x::RealOrComplexArray)
  if vecnorm(x) > f.r return +Inf end
  return 0.0
end

function prox(f::IndBallL2, gamma::Float64, x::RealOrComplexArray)
  y = x*min(1, f.r/vecnorm(x))
  return y, 0.0
end

fun_name(f::IndBallL2) = "indicator of an L2 norm ball"
fun_type(f::IndBallL2) = "C^n → R ∪ {+∞}"
fun_expr(f::IndBallL2) = "x ↦ 0 if ||x|| ⩽ r, +∞ otherwise"
fun_params(f::IndBallL2) = "r = $(f.r)"
