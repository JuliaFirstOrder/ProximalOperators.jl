# indicator of a generic box

export IndBox, IndBallLinf

"""
**Indicator of a box**

	IndBox(low, up)

Returns the indicator function of the set
```math
S = \\{ x : low \\leq x \\leq up \\}.
```
Parameters `low` and `up` can be either scalars or arrays of the same dimension as the space: they must satisfy `low <= up`, and are allowed to take values `-Inf` and `+Inf` to indicate unbounded coordinates.
"""
struct IndBox{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
	lb::T
	ub::S
	function IndBox{T,S}(lb::T, ub::S) where {T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}}
		if !(eltype(lb) <: Real && eltype(ub) <: Real)
			error("`lb` and `ub` must be real")
		end
		if any(lb .> ub)
			error("`lb` and `ub` must satisfy `lb <= ub`")
		else
			new(lb, ub)
		end
	end
end

is_separable(f::IndBox) = true
is_convex(f::IndBox) = true
is_set(f::IndBox) = true
is_cone(f::IndBox) = all((f.lb .== -Inf) .+ (f.ub .== +Inf) .> 0)

IndBox(lb::T, ub::T) where {T <: Real} = IndBox{T, T}(lb, ub)

IndBox(lb::T, ub::S) where {T <: AbstractArray, S <: Real} = IndBox{T, S}(lb, ub)

IndBox(lb::T, ub::S) where {T <: Real, S <: AbstractArray} = IndBox{T, S}(lb, ub)

IndBox(lb::T, ub::S) where {T <: AbstractArray, S <: AbstractArray} =
	size(lb) != size(ub) ? error("bounds must have the same dimensions, or at least one of them be scalar") :
	IndBox{T, S}(lb, ub)

function (f::IndBox)(x::AbstractArray{R}) where R <: Real
	for k in eachindex(x)
		if x[k] < get_kth_elem(f.lb, k) || x[k] > get_kth_elem(f.ub, k)
			return +Inf
		end
	end
	return 0.0
end

function prox!(y::AbstractArray{R}, f::IndBox, x::AbstractArray{R}, gamma::Real=one(R)) where R <: Real
	for k in eachindex(x)
		if x[k] < get_kth_elem(f.lb, k)
			y[k] = get_kth_elem(f.lb, k)
		elseif x[k] > get_kth_elem(f.ub, k)
			y[k] = get_kth_elem(f.ub, k)
		else
			y[k] = x[k]
		end
	end
	return zero(R)
end

prox!(y::AbstractArray{R}, f::IndBox, x::AbstractArray{R}, gamma::AbstractArray) where {R <: Real} = prox!(y, f, x, one(R))

"""
**Indicator of a ``L_∞`` norm ball**

	IndBallLinf(r=1.0)

Returns the indicator function of the set
```math
S = \\{ x : \\max (|x_i|) \\leq r \\}.
```
Parameter `r` must be positive.
"""
IndBallLinf(r::R=1.0) where {R <: Real} = IndBox(-r, r)

fun_name(f::IndBox) = "indicator of a box"
fun_dom(f::IndBox) = "AbstractArray{Real}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
	string( "lb = ", typeof(f.lb) <: AbstractArray ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
			"ub = ", typeof(f.ub) <: AbstractArray ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)

function prox_naive(f::IndBox, x::AbstractArray{R}, gamma=one(R)) where R <: Real
	y = min.(f.ub, max.(f.lb, x))
	return y, zero(R)
end
