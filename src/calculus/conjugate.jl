# Conjugate

export Conjugate

"""
**Convex conjugate**

    Conjugate(f)

Returns the convex conjugate (also known as Fenchel conjugate, or Fenchel-Legendre transform) of function `f`, that is
```math
f^*(x) = \\sup_y \\{ \\langle y, x \\rangle - f(y) \\}.
```
"""
struct Conjugate{T}
    f::T
    function Conjugate{T}(f::T) where T
        if is_convex(f) == false
            error("`f` must be convex")
        end
        new(f)
    end
end

is_prox_accurate(::Type{Conjugate{T}}) where T = is_prox_accurate(T)
is_convex(::Type{Conjugate{T}}) where T = true
is_cone(::Type{Conjugate{T}}) where T = is_cone(T) && is_convex(T)
is_smooth(::Type{Conjugate{T}}) where T = is_strongly_convex(T)
is_strongly_convex(::Type{Conjugate{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{Conjugate{T}}) where T = is_generalized_quadratic(T)
is_set(::Type{Conjugate{T}}) where T = is_convex(T) && is_support(T)
is_positively_homogeneous(::Type{Conjugate{T}}) where T = is_convex(T) && is_set(T)

fun_dom(f::Conjugate) = fun_dom(f.f)

Conjugate(f::T) where T = Conjugate{T}(f)

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!(y::AbstractArray{R}, g::Conjugate, x::AbstractArray{R}, gamma) where R
    # Moreau identity
    v = prox!(y, g.f, x/gamma, 1/gamma)
    if is_set(g)
        v = R(0)
    else
        v = dot(x, y) - gamma * dot(y, y) - v
    end
    y .= x .- gamma .* y
    return v
end

# complex case, need to cast inner products to real

function prox!(y::AbstractArray{Complex{R}}, g::Conjugate, x::AbstractArray{Complex{R}}, gamma) where R
    v = prox!(y, g.f, x/gamma, 1/gamma)
    if is_set(g)
        v = R(0)
    else
        v = real(dot(x, y)) - gamma * real(dot(y, y)) - v
    end
    y .= x .- gamma .* y
    return v
end

# naive implementation

function prox_naive(g::Conjugate, x::AbstractArray{T}, gamma) where {R, T <: RealOrComplex{R}}
    y, v = prox_naive(g.f, x/gamma, 1/gamma)
    return x - gamma * y, if is_set(g) R(0) else real(dot(x, y)) - gamma * real(dot(y, y)) - v end
end

# TODO: hard-code conjugation rules? E.g. precompose/epicompose
