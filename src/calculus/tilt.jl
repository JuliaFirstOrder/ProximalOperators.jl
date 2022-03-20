# Tilting (addition with affine function)

export Tilt

"""
    Tilt(f, a, b=0.0)

Given function `f`, an array `a` and a constant `b` (optional), return function
```math
g(x) = f(x) + \\langle a, x \\rangle + b.
```
"""
struct Tilt{T, S, R}
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

Tilt(f::T, a::S) where {T, S} = Tilt{T, S, real(eltype(S))}(f, a, real(eltype(S))(0))

function (g::Tilt)(x)
    return g.f(x) + real(dot(g.a, x)) + g.b
end

function prox!(y, g::Tilt, x, gamma)
    v = prox!(y, g.f, x .- gamma .* g.a, gamma)
    return v + real(dot(g.a, y)) + g.b
end

function prox_naive(g::Tilt, x, gamma)
    y, v = prox_naive(g.f, x .- gamma .* g.a, gamma)
    return y, v + real(dot(g.a, y)) + g.b
end
