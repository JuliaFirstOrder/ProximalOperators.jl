# indicator of the Cartesian product of real binary sets

export IndBinary

"""
**Indicator of the product of binary sets**

    IndBinary(low, up)

Returns the indicator function of the set
```math
S = \\{ x : x_i = low_i\\ \\text{or}\\ x_i = up_i \\},
```
Parameters `low` and `up` can be either scalars or arrays of the same dimension as the space.
"""

struct IndBinary{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  low::T
  high::S
end

is_set(f::IndBinary) = true

IndBinary(low::T=0.0, high::T=1.0) where {T <: Real} = IndBinary{T, T}(low, high)

IndBinary_low(f::IndBinary{T, S}, i) where {T <: Real, S} = f.low
IndBinary_low(f::IndBinary{T, S}, i) where {T <: AbstractArray, S} = f.low[i]
IndBinary_high(f::IndBinary{T, S}, i) where {T, S <: Real} = f.high
IndBinary_high(f::IndBinary{T, S}, i) where {T, S <: AbstractArray} = f.high[i]

function (f::IndBinary)(x::AbstractArray{T}) where T <: Real
  for k in eachindex(x)
    if x[k] != IndBinary_low(f, k) && x[k] != IndBinary_high(f, k)
      return +Inf
    end
  end
  return zero(T)
end

function prox!(y::AbstractArray{T}, f::IndBinary, x::AbstractArray{T}, gamma::Real=1.0) where T <: Real
  low = zero(T)
  high = zero(T)
  for k in eachindex(x)
    low = IndBinary_low(f, k)
    high = IndBinary_high(f, k)
    if abs(x[k] - low) < abs(x[k] - high)
      y[k] = low
    else
      y[k] = high
    end
  end
  return zero(T)
end

fun_name(f::IndBinary) = "indicator of binary array"
fun_dom(f::IndBinary) = "AbstractArray{Real}"

function prox_naive(f::IndBinary, x::AbstractArray{T}, gamma::Real=1.0) where T <: Real
  distlow = abs.(x .- f.low)
  disthigh = abs.(x .- f.high)
  indlow = distlow .< disthigh
  indhigh = distlow .>= disthigh
  y = f.low.*indlow + f.high.*indhigh
  return y, 0.0
end
