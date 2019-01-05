# indicator of the zero cone

export IndZero

"""
**Indicator of the zero cone**

	IndZero()

Returns the indicator function of the set containing the origin, the "zero cone".
"""
struct IndZero <: ProximableFunction end

is_separable(f::IndZero) = true
is_convex(f::IndZero) = true
is_singleton(f::IndZero) = true
is_cone(f::IndZero) = true
is_affine(f::IndZero) = true

function (f::IndZero)(x::AbstractArray{C}) where {R <: Real, C <: Union{R, Complex{R}}}
	for k in eachindex(x)
		if x[k] != zero(C)
			return R(Inf)
		end
	end
	return zero(R)
end

function prox!(y::AbstractArray{C}, f::IndZero, x::AbstractArray{C}, gamma=one(R)) where {R <: Real, C <: Union{R, Complex{R}}}
	for k in eachindex(x)
		y[k] = zero(C)
	end
	return zero(R)
end

fun_name(f::IndZero) = "indicator of the zero cone"
fun_dom(f::IndZero) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndZero) = "x ↦ 0 if all(x = 0), +∞ otherwise"
fun_params(f::IndZero) = "none"

function prox_naive(f::IndZero, x::AbstractArray{C}, gamma=one(R)) where {R <: Real, C <: Union{R, Complex{R}}}
	return zero(x), zero(R)
end
