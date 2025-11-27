export Translate

"""
    Translate(f, b)

Return the translated function
```math
g(x) = f(x + b)
```
"""
struct Translate{T, V}
    f::T
    b::V
end

is_separable(::Type{<:Translate{T}}) where T = is_separable(T)
is_proximable(::Type{<:Translate{T}}) where T = is_proximable(T)
is_convex(::Type{<:Translate{T}}) where T = is_convex(T)
is_set_indicator(::Type{<:Translate{T}}) where T = is_set_indicator(T)
is_singleton_indicator(::Type{<:Translate{T}}) where T = is_singleton_indicator(T)
is_cone_indicator(::Type{<:Translate{T}}) where T = is_cone_indicator(T)
is_affine_indicator(::Type{<:Translate{T}}) where T = is_affine_indicator(T)
is_smooth(::Type{<:Translate{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Translate{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Translate{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Translate{T}}) where T = is_strongly_convex(T)

function (g::Translate)(x)
    return g.f(x .+ g.b)
end

function gradient!(y, g::Translate, x)
    z = x .+ g.b
    v = gradient!(y, g.f, z)
    return v
end

function prox!(y, g::Translate, x, gamma)
    z = x .+ g.b
    v = prox!(y, g.f, z, gamma)
    y .-= g.b
    return v
end

function prox_naive(g::Translate, x, gamma)
    y, v = prox_naive(g.f, x .+ g.b, gamma)
    return y - g.b, v
end
