# indicator of the ball of matrices with (at most) a given rank

"""
  IndBallRank(r::Int64)

Returns the function `g = ind{X : rank(X) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallRank <: IndicatorFunction
  r::Int64
  IndBallRank(r::Int64) =
    r <= 0 ? error("parameter r must be a positive integer") : new(r)
end

function call(f::IndBallRank, x::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r return 0.0 end
  u, s, v = svds(x, nsv=f.r+1)
  # the tolerance in the following line should be customizable
  if s[end]/s[1] <= 1e-14 return 0.0 end
  return +Inf
end

function prox(f::IndBallRank, gamma::Float64, x::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r return (x, 0.0) end
  u, s, v = svds(x, nsv=f.r)
  return (u*diagm(s))*v', 0.0
end

fun_name(f::IndBallRank) = "indicator of the set of rank-r matrices"
fun_type(f::IndBallRank) = "C^{n×m} → R ∪ {+∞}"
fun_expr(f::IndBallRank) = "x ↦ 0 if rank(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallRank) = "r = $(f.r)"
