export Translate

"""
**Translation**

    Translate(f, b)

Returns the translated function
```math
g(x) = f(x + b)
```
"""
struct Translate{T, V <: Union{Number, AbstractArray, Tuple}}
    f::T
    b::V
end

is_separable(::Type{<:Translate{T}}) where T = is_separable(T)
is_prox_accurate(::Type{<:Translate{T}}) where T = is_prox_accurate(T)
is_convex(::Type{<:Translate{T}}) where T = is_convex(T)
is_set(::Type{<:Translate{T}}) where T = is_set(T)
is_singleton(::Type{<:Translate{T}}) where T = is_singleton(T)
is_cone(::Type{<:Translate{T}}) where T = is_cone(T)
is_affine(::Type{<:Translate{T}}) where T = is_affine(T)
is_smooth(::Type{<:Translate{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Translate{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Translate{T}}) where T = is_strongly_convex(T)

function (g::Translate)(x::T) where {T <: Union{Tuple, AbstractArray}}
    return g.f(x .+ g.b)
end

function gradient!(y, g::Translate, x)
    z = x .+ g.b
    v = gradient!(y, g.f, z)
    return v
end

function prox!(y, g::Translate, x, gamma=1.0)
    z = x .+ g.b
    v = prox!(y, g.f, z, gamma)
    y .-= g.b
    return v
end

function prox_naive(g::Translate, x, gamma=1.0)
    y, v = prox_naive(g.f, x .+ g.b, gamma)
    return y - g.b, v
end

fun_name(f::Translate) = "Translation"
fun_dom(f::Translate) = fun_dom(f.f)
fun_expr(f::Translate) = "x ↦ f(x + b)"
