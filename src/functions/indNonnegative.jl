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
	return R(0)
end

function prox!(y::AbstractArray{R}, f::IndNonnegative, x::AbstractArray{R}, gamma=R(1)) where R <: Real
	for k in eachindex(x)
		if x[k] < 0
			y[k] = R(0)
		else
			y[k] = x[k]
		end
	end
	return R(0)
end

fun_name(f::IndNonnegative) = "indicator of the Nonnegative cone"
fun_dom(f::IndNonnegative) = "AbstractArray{Real}"
fun_expr(f::IndNonnegative) = "x ↦ 0 if all(0 ⩽ x), +∞ otherwise"
fun_params(f::IndNonnegative) = "none"

function prox_naive(f::IndNonnegative, x::AbstractArray{R}, gamma=R(1)) where R <: Real
	y = max.(R(0), x)
	return y, R(0)
end
