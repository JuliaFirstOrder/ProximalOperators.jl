# indicator of the ball of matrices with (at most) a given rank

"""
  IndBallRank(r::Int=1)

Returns the function `g = ind{X : rank(X) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallRank{I <: Integer} <: IndicatorFunction
  r::I
  function IndBallRank(r::I)
    if r <= 0
      error("parameter r must be a positive integer")
    else
      new(r)
    end
  end
end

IndBallRank{I <: Integer}(r::I=1) = IndBallRank{I}(r)

function (f::IndBallRank){T <: RealOrComplex}(x::AbstractArray{T,2})
  maxr = minimum(size(x))
  if maxr <= f.r return 0.0 end
  svdobj = svds(x, nsv=f.r+1)[1]
  # the tolerance in the following line should be customizable
  if svdobj[:S][end]/svdobj[:S][1] <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: Real}(f::IndBallRank, x::AbstractArray{T,2}, y::AbstractArray{T,2}, gamma::Real=1.0)
  maxr = minimum(size(x))
  if maxr <= f.r
    y[:] = x
    return 0.0
  end
  svdobj = svds(x, nsv=f.r)[1]
  for i = 1:size(x,1)
    for j = 1:size(x,2)
      y[i,j] = 0.0
      for k = 1:f.r
        y[i,j] += svdobj[:U][i,k]*svdobj[:S][k]*svdobj[:Vt][j,k]
      end
    end
  end
  return 0.0
end

function prox!{T <: Complex}(f::IndBallRank, x::AbstractArray{T,2}, y::AbstractArray{T,2}, gamma::Real=1.0)
  maxr = minimum(size(x))
  if maxr <= f.r
    y[:] = x
    return 0.0
  end
  svdobj = svds(x, nsv=f.r)[1]
  for i = 1:size(x,1)
    for j = 1:size(x,2)
      y[i,j] = 0.0
      for k = 1:f.r
        y[i,j] += svdobj[:U][i,k]*svdobj[:S][k]*conj(svdobj[:Vt][j,k])
      end
    end
  end
  return 0.0
end

fun_name(f::IndBallRank) = "indicator of the set of rank-r matrices"
fun_dom(f::IndBallRank) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::IndBallRank) = "x ↦ 0 if rank(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallRank) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndBallRank, x::AbstractArray{T,2}, gamma::Real=1.0)
  maxr = minimum(size(x))
  if maxr <= f.r
    y = x
    return y, 0.0
  end
  U, S, V = svd(x)
  M = U[:,1:f.r]*spdiagm(S[1:f.r])
  y = M*V[:,1:f.r]'
  return y, 0.0
end
