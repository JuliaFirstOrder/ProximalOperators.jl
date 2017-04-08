# indicator of the L0 norm ball with given (integer) radius

"""
  IndBallL0(r::Int=1)

Returns the function `g = ind{x : countnz(x) ⩽ r}`, for an integer parameter `r > 0`.
"""

immutable IndBallL0{I <: Integer} <: ProximableFunction
  r::I
  function IndBallL0(r::I)
    if r <= 0
      error("parameter r must be a positive integer")
    else
      new(r)
    end
  end
end

is_set(f::IndBallL0) = true

IndBallL0{I <: Integer}(r::I) = IndBallL0{I}(r)

function (f::IndBallL0){T <: RealOrComplex}(x::AbstractArray{T})
  if countnz(x) > f.r
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, f::IndBallL0, x::AbstractArray{T}, gamma::Real=1.0)
  p = []
  if ndims(x) == 1
    p = selectperm(x, 1:f.r, by=abs, rev=true)
  else
    p = selectperm(x[:], 1:f.r, by=abs, rev=true)
  end
  sort!(p)
  idx = 1
  for i = 1:length(p)
    y[idx:p[i]-1] = 0.0
    y[p[i]] = x[p[i]]
    idx = p[i]+1
  end
  y[idx:end] = 0.0
  return 0.0
end

fun_name(f::IndBallL0) = "indicator of an L0 pseudo-norm ball"
fun_dom(f::IndBallL0) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndBallL0) = "x ↦ 0 if countnz(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallL0) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndBallL0, x::AbstractArray{T}, gamma::Real=1.0)
  p = sortperm(abs.(x)[:], rev=true)
  y = similar(x)
  y[p[1:f.r]] = x[p[1:f.r]]
  y[p[f.r+1:end]] = 0.0
  return y, 0.0
end
