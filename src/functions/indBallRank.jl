# indicator of the ball of matrices with (at most) a given rank

"""
  IndBallRank(r::Int=1)

Returns the function `g = ind{X : rank(X) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallRank <: IndicatorFunction
  r::Int
  IndBallRank(r::Int=1) =
    r <= 0 ? error("parameter r must be a positive integer") : new(r)
end

if VERSION >= v"0.5.0-rc1"

@compat function (f::IndBallRank)(x::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r return 0.0 end
  svdobj = svds(x, nsv=f.r+1)[1]
  # the tolerance in the following line should be customizable
  if svdobj[:S][end]/svdobj[:S][1] <= 1e-14 return 0.0 end
  return +Inf
end

function prox!(f::IndBallRank, x::RealOrComplexMatrix, gamma::Float64, y::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r
    y[:] = x
    return 0.0
  end
  svdobj = svds(x, nsv=f.r)[1]
  y[:] = (svdobj[:U].*svdobj[:S]')*svdobj[:Vt]'
  return 0.0
end

else

@compat function (f::IndBallRank)(x::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r return 0.0 end
  u, s, v = svds(x, nsv=f.r+1)
  # the tolerance in the following line should be customizable
  if s[end]/s[1] <= 1e-14 return 0.0 end
  return +Inf
end

function prox!(f::IndBallRank, x::RealOrComplexMatrix, gamma::Float64, y::RealOrComplexMatrix)
  maxr = minimum(size(x))
  if maxr <= f.r return (x, 0.0) end
  u, s, v = svds(x, nsv=f.r)
  y[:] = (u.*s')*v'
  return 0.0
end

end

fun_name(f::IndBallRank) = "indicator of the set of rank-r matrices"
fun_type(f::IndBallRank) = "Array{Complex,2} → Real ∪ {+∞}"
fun_expr(f::IndBallRank) = "x ↦ 0 if rank(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallRank) = "r = $(f.r)"

function prox_naive(f::IndBallRank, x::RealOrComplexMatrix, gamma::Float64=1.0)
  maxr = minimum(size(x))
  if maxr <= f.r
    y = x
    return 0.0
  end
  U, S, V = svd(x)
  M = U[:,1:f.r]*spdiagm(S[1:f.r])
  y = M*V[:,1:f.r]'
  return y, 0.0
end
