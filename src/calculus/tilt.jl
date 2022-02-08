# Tilting (addition with affine function)

export Tilt

"""
    Tilt(f, a, b=0.0)

Given function `f`, an array `a` and a constant `b` (optional), return function
```math
g(x) = f(x) + \\langle a, x \\rangle + b.
```
"""
struct Tilt{T, S <: AbstractArray, R <: Real}
    f::T
    a::S
    b::R
end

is_separable(::Type{<:Tilt{T}}) where T = is_separable(T)
is_prox_accurate(::Type{<:Tilt{T}}) where T = is_prox_accurate(T)
is_convex(::Type{<:Tilt{T}}) where T = is_convex(T)
is_singleton(::Type{<:Tilt{T}}) where T = is_singleton(T)
is_smooth(::Type{<:Tilt{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Tilt{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Tilt{T}}) where T = is_strongly_convex(T)

Tilt(f::T, a::S) where {R <: Real, T, S <: AbstractArray{R}} = Tilt{T, S, R}(f, a, R(0))

function (g::Tilt)(x::AbstractArray{T}) where T <: RealOrComplex
    return g.f(x) + dot(g.a, x) + g.b
end

function prox!(y::AbstractArray{T}, g::Tilt, x::AbstractArray{T}, gamma=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    v = prox!(y, g.f, x .- gamma .* g.a, gamma)
    return v + dot(g.a, y) + g.b
end

fun_name(f::Tilt) = string("Tilted ", fun_name(f.f))
fun_dom(f::Tilt) = fun_dom(f.f)
fun_expr(f::Tilt) = string(fun_expr(f.f)," + a'x + b")
fun_params(f::Tilt) = "a = $(typeof(f.a)), b = $(f.b)"

function prox_naive(g::Tilt, x::AbstractArray{T}, gamma=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    y, v = prox_naive(g.f, x .- gamma .* g.a, gamma)
    return y, v + dot(g.a, y) + g.b
end
