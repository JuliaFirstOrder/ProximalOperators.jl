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
struct IndBinary{C <: RealOrComplex, T <: Union{C, AbstractArray{C}}, S <: Union{C, AbstractArray{C}}} <: ProximableFunction
	low::T
	high::S
end

is_set(f::IndBinary) = true

IndBinary() = IndBinary(0.0, 1.0)

IndBinary_low(f::IndBinary{C, C, S}, i) where {C, S} = f.low
IndBinary_low(f::IndBinary{C, T, S}, i) where {C, T <: AbstractArray, S} = f.low[i]
IndBinary_high(f::IndBinary{C, T, C}, i) where {C, T} = f.high
IndBinary_high(f::IndBinary{C, T, S}, i) where {C, T, S <: AbstractArray} = f.high[i]

function (f::IndBinary)(x::AbstractArray{C}) where {R <: Real, C <: RealOrComplex{R}}
	for k in eachindex(x)
		if x[k] != IndBinary_low(f, k) && x[k] != IndBinary_high(f, k)
			return R(Inf)
		end
	end
	return R(0)
end

function prox!(y::AbstractArray{T}, f::IndBinary, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: Union{R, Complex{R}}}
	for k in eachindex(x)
		low = IndBinary_low(f, k)
		high = IndBinary_high(f, k)
		if abs(x[k] - low) < abs(x[k] - high)
			y[k] = low
		else
			y[k] = high
		end
	end
	return R(0)
end

fun_name(f::IndBinary) = "indicator of binary array"
fun_dom(f::IndBinary) = "AbstractArray{Real}"

function prox_naive(f::IndBinary, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: Union{R, Complex{R}}}
	distlow = abs.(x .- f.low)
	disthigh = abs.(x .- f.high)
	indlow = distlow .< disthigh
	indhigh = distlow .>= disthigh
	y = f.low.*indlow + f.high.*indhigh
	return y, R(0)
end
