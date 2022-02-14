# Postcompose with scaling and sum

export Postcompose

"""
    Postcompose(f, a=1, b=0)

Return the function
```math
g(x) = a\\cdot f(x) + b.
```
"""
struct Postcompose{T, R, S}
    f::T
    a::R
    b::S
    function Postcompose{T,R,S}(f::T, a::R, b::S) where {T, R, S}
        if a <= 0
            error("parameter `a` must be positive")
        else
            new(f, a, b)
        end
    end
end

is_prox_accurate(::Type{<:Postcompose{T}}) where T = is_prox_accurate(T)
is_separable(::Type{<:Postcompose{T}}) where T = is_separable(T)
is_convex(::Type{<:Postcompose{T}}) where T = is_convex(T)
is_set(::Type{<:Postcompose{T}}) where T = is_set(T)
is_singleton(::Type{<:Postcompose{T}}) where T = is_singleton(T)
is_cone(::Type{<:Postcompose{T}}) where T = is_cone(T)
is_affine(::Type{<:Postcompose{T}}) where T = is_affine(T)
is_smooth(::Type{<:Postcompose{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Postcompose{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Postcompose{T}}) where T = is_strongly_convex(T)

Postcompose(f::T, a::R=1, b::S=0) where {T, R <: Real, S <: Real} = Postcompose{T, R, S}(f, a, b)

Postcompose(f::Postcompose{T, R, S}, a::R=1, b::S=0) where {T, R <: Real, S <: Real} = Postcompose{T, R, S}(f.f, a * f.a, b + a * f.b)

function (g::Postcompose)(x)
    return g.a * g.f(x) + g.b
end

function gradient!(y, g::Postcompose, x)
    v = gradient!(y, g.f, x)
    y .*= g.a
    return g.a * v + g.b
end

function prox!(y, g::Postcompose, x, gamma)
    v = prox!(y, g.f, x, g.a * gamma)
    return g.a * v + g.b
end

function prox_naive(g::Postcompose, x, gamma)
    y, v = prox_naive(g.f, x, g.a * gamma)
    return y, g.a * v + g.b
end
