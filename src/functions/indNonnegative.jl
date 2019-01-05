# indicator of nonnegative orthant

export IndNonnegative

"""
**Indicator of the nonnegative orthant**

	IndNonnegative()

Returns the indicator of the set
```math
C = \\{ x : x \\geq 0 \\}.
```
"""
struct IndNonnegative <: ProximableFunction end

is_separable(f::IndNonnegative) = true
is_convex(f::IndNonnegative) = true
is_cone(f::IndNonnegative) = true

function (f::IndNonnegative)(x::AbstractArray{R}) where R <: Real
	for k in eachindex(x)
		if x[k] < 0
			return R(Inf)
		end
	end
	return zero(R)
end

function prox!(y::AbstractArray{R}, f::IndNonnegative, x::AbstractArray{R}, gamma=one(R)) where R <: Real
	for k in eachindex(x)
		if x[k] < 0
			y[k] = zero(R)
		else
			y[k] = x[k]
		end
	end
	return zero(R)
end

fun_name(f::IndNonnegative) = "indicator of the Nonnegative cone"
fun_dom(f::IndNonnegative) = "AbstractArray{Real}"
fun_expr(f::IndNonnegative) = "x ↦ 0 if all(0 ⩽ x), +∞ otherwise"
fun_params(f::IndNonnegative) = "none"

function prox_naive(f::IndNonnegative, x::AbstractArray{R}, gamma=one(R)) where R <: Real
	y = max.(zero(R), x)
	return y, zero(R)
end
