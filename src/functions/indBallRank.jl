# indicator of the ball of matrices with (at most) a given rank

export IndBallRank

"""
**Indicator of rank ball**

    IndBallRank(r=1)

Returns the indicator function of the set of matrices of rank at most `r`:
```math
S = \\{ X : \\mathrm{rank}(X) \\leq r \\},
```
Parameter `r` must be a positive integer.
"""

immutable IndBallRank{I <: Integer} <: ProximableFunction
  r::I
  function IndBallRank{I}(r::I) where {I <: Integer}
    if r <= 0
      error("parameter r must be a positive integer")
    else
      new(r)
    end
  end
end

is_set(f::IndBallRank) = false

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

function prox!(y::AbstractMatrix{T}, f::IndBallRank, x::AbstractMatrix{T}, gamma::R=1.0) where {R <: Real, T <: Union{R, Complex{R}}}
  maxr = minimum(size(x))
  if maxr <= f.r
    y[:] = x
    return 0.0
  end
  svdobj = svds(x, nsv=f.r)[1]
  M = svdobj[:S][1:f.r] .* svdobj[:Vt][1:f.r,:]
  A_mul_B!(y, svdobj[:U][:,1:f.r], M)
  return 0.0
end

fun_name(f::IndBallRank) = "indicator of the set of rank-r matrices"
fun_dom(f::IndBallRank) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::IndBallRank) = "x ↦ 0 if rank(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallRank) = "r = $(f.r)"

function prox_naive(f::IndBallRank, x::AbstractMatrix{T}, gamma=1.0) where T
  maxr = minimum(size(x))
  if maxr <= f.r
    y = x
    return y, 0.0
  end
  U, S, V = svd(x)
  y = U[:,1:f.r]*(spdiagm(S[1:f.r])*V[:,1:f.r]')
  return y, 0.0
end
