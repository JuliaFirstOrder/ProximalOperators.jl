# indicator of the L0 norm ball with given (integer) radius

"""
  IndBallL0(r::Int64)

Returns the function `g = ind{x : countnz(x) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallL0 <: IndicatorFunction
  r::Int64
  IndBallL0(r::Int64) =
    r <= 0 ? error("parameter r must be a positive integer") : new(r)
end

function call(f::IndBallL0, x::RealOrComplexArray)
  if countnz(x) > f.r return +Inf end
  return 0.0
end

function prox(f::IndBallL0, x::RealOrComplexArray, gamma::Float64=1.0)
  y = zeros(x)
  if f.r < log2(length(x))
    p = selectperm(abs(x)[:], 1:f.r, rev=true)
    y[p] = x[p]
  else
    p = sortperm(abs(x)[:], rev=true)
    y[p[1:f.r]] = x[p[1:f.r]]
  end
  return y, 0.0
end

fun_name(f::IndBallL0) = "indicator of an L0 pseudo-norm ball"
fun_type(f::IndBallL0) = "C^n → R ∪ {+∞}"
fun_expr(f::IndBallL0) = "x ↦ 0 if countnz(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallL0) = "r = $(f.r)"
