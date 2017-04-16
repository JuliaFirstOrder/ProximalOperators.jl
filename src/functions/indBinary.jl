# indicator of the Cartesian product of real binary sets

"""
  IndBinary(low, high)

Returns the function `g = ind{x : x_i == low || x_i == high}`.
"""

immutable IndBinary{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  low::T
  high::S
end

is_set(f::IndBinary) = true

IndBinary{T <: Real}(low::T=0.0, high::T=1.0) = IndBinary{T, T}(low, high)

IndBinary_low{T <: Real, S}(f::IndBinary{T, S}, i) = f.low
IndBinary_low{T <: AbstractArray, S}(f::IndBinary{T, S}, i) = f.low[i]
IndBinary_high{T, S <: Real}(f::IndBinary{T, S}, i) = f.high
IndBinary_high{T, S <: AbstractArray}(f::IndBinary{T, S}, i) = f.high[i]

function (f::IndBinary){T <: Real}(x::AbstractArray{T})
  for k in eachindex(x)
    if x[k] != IndBinary_low(f, k) && x[k] != IndBinary_high(f, k)
      return +Inf
    end
  end
  return 0.0
end

function prox!{T <: Real}(y::AbstractArray{T}, f::IndBinary, x::AbstractArray{T}, gamma::Real=1.0)
  low = 0.0
  high = 0.0
  for k in eachindex(x)
    low = IndBinary_low(f, k)
    high = IndBinary_high(f, k)
    if abs(x[k] - low) < abs(x[k] - high)
      y[k] = low
    else
      y[k] = high
    end
  end
  return 0.0
end

fun_name(f::IndBinary) = "indicator of binary array"
fun_dom(f::IndBinary) = "AbstractArray{Real}"

function prox_naive{T <: Real}(f::IndBinary, x::AbstractArray{T}, gamma::Real=1.0)
  distlow = abs.(x-f.low)
  disthigh = abs.(x-f.high)
  indlow = distlow .< disthigh
  indhigh = distlow .>= disthigh
  y = f.low.*indlow + f.high.*indhigh
  return y, 0.0
end
