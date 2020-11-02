# Postcompose with scaling and sum

export Postcompose

"""
**Postcomposition with an affine transformation**

    Postcompose(f, a=1, b=0)

Returns the function
```math
g(x) = a\\cdot f(x) + b.
```
"""
struct Postcompose{T <: ProximableFunction, R <: Real, S <: Real} <: ProximableFunction
    f::T
    a::R
    b::S
    function Postcompose{T,R,S}(f::T, a::R, b::S) where {T <: ProximableFunction, R <: Real, S <: Real}
        if a <= 0
            error("parameter `a` must be positive")
        else
            new(f, a, b)
        end
    end
end

is_prox_accurate(f::Postcompose) = is_prox_accurate(f.f)
is_separable(f::Postcompose) = is_separable(f.f)
is_convex(f::Postcompose) = is_convex(f.f)
is_set(f::Postcompose) = is_set(f.f)
is_singleton(f::Postcompose) = is_singleton(f.f)
is_cone(f::Postcompose) = is_cone(f.f)
is_affine(f::Postcompose) = is_affine(f.f)
is_smooth(f::Postcompose) = is_smooth(f.f)
is_quadratic(f::Postcompose) = is_quadratic(f.f)
is_generalized_quadratic(f::Postcompose) = is_generalized_quadratic(f.f)
is_strongly_convex(f::Postcompose) = is_strongly_convex(f.f)

Postcompose(f::T, a::R=1, b::S=0) where {T <: ProximableFunction, R <: Real, S <: Real} = Postcompose{T, R, S}(f, a, b)

Postcompose(f::Postcompose{T, R, S}, a::R=1, b::S=0) where {T <: ProximableFunction, R <: Real, S <: Real} = Postcompose{T, R, S}(f.f, a * f.a, b + a * f.b)

function (g::Postcompose)(x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    return g.a * g.f(x) + g.b
end

function gradient!(y::AbstractArray{T}, g::Postcompose, x::AbstractArray{T}) where {
    R <: Real, T <: RealOrComplex{R}
}
    v = gradient!(y, g.f, x)
    y .*= g.a
    return g.a * v + g.b
end

function prox!(y::AbstractArray{T}, g::Postcompose, x::AbstractArray{T}, gamma=R(1)) where {
    R <: Real, T <: RealOrComplex{R}
}
    v = prox!(y, g.f, x, g.a * gamma)
    return g.a * v + g.b
end

function prox_naive(g::Postcompose, x::AbstractArray{T}, gamma=R(1)) where {
    R <: Real, T <: RealOrComplex{R}
}
    y, v = prox_naive(g.f, x, g.a * gamma)
    return y, g.a * v + g.b
end

fun_name(f::Postcompose) = string("Postcomposition of ", fun_name(f.f))
fun_dom(f::Postcompose) = fun_dom(f.f)
fun_expr(f::Postcompose) = "x ↦ a*f(x)+b"
fun_params(f::Postcompose) = string("f(x) = ", fun_expr(f.f), ", a = $(f.a), b = $(f.b)")
