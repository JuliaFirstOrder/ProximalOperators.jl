# Tilting (addition with affine function)

export Tilt

"""
**Linear tilting**

	Tilt(f, a, b=0.0)

Given function `f`, an array `a` and a constant `b` (optional), returns function
```math
g(x) = f(x) + \\langle a, x \\rangle + b.
```
"""
struct Tilt{T <: ProximableFunction, S <: AbstractArray, R <: Real} <: ProximableFunction
	f::T
	a::S
	b::R
end

is_separable(f::Tilt) = is_separable(f.f)
is_prox_accurate(f::Tilt) = is_prox_accurate(f.f)
is_convex(f::Tilt) = is_convex(f.f)
is_singleton(f::Tilt) = is_singleton(f.f)
is_smooth(f::Tilt) = is_smooth(f.f)
is_quadratic(f::Tilt) = is_quadratic(f.f)
is_generalized_quadratic(f::Tilt) = is_generalized_quadratic(f.f)
is_strongly_convex(f::Tilt) = is_strongly_convex(f.f)

Tilt(f::T, a::S) where {R <: Real, T <: ProximableFunction, S <: AbstractArray{R}} = Tilt{T, S, R}(f, a, R(0))

function (g::Tilt)(x::AbstractArray{T}) where T <: RealOrComplex
	return g.f(x) + dot(g.a, x) + g.b
end

function prox!(y::AbstractArray{T}, g::Tilt, x::AbstractArray{T}, gamma=one(R)) where {R <: Real, T <: RealOrComplex{R}}
	v = prox!(y, g.f, x .- gamma .* g.a, gamma)
	return v + dot(g.a, y) + g.b
end

fun_name(f::Tilt) = string("Tilted ", fun_name(f.f))
fun_dom(f::Tilt) = fun_dom(f.f)
fun_expr(f::Tilt) = string(fun_expr(f.f)," + a'x + b")
fun_params(f::Tilt) = "a = $(typeof(f.a)), b = $(f.b)"

function prox_naive(g::Tilt, x::AbstractArray{T}, gamma=one(R)) where {R <: Real, T <: RealOrComplex{R}}
	y, v = prox_naive(g.f, x .- gamma .* g.a, gamma)
	return y, v + dot(g.a, y) + g.b
end
