# indicator of a point

export IndPoint

"""
**Indicator of a singleton**

    IndPoint(p=0.0)

Returns the indicator of the set
```math
C = \\{p \\}.
```
Parameter `p` can be a scalar, in which case the unique element of `S` has uniform coefficients.
"""
struct IndPoint{T}
    p::T
    function IndPoint{T}(p::T) where {T}
        new(p)
    end
end

is_separable(f::Type{<:IndPoint}) = true
is_convex(f::Type{<:IndPoint}) = true
is_singleton(f::Type{<:IndPoint}) = true
is_affine(f::Type{<:IndPoint}) = true

IndPoint(p::T=0.0) where T = IndPoint{T}(p)

function (f::IndPoint)(x::AbstractArray{C}) where {R <: Real, C <: Union{R, Complex{R}}}
    if all(x .≈ f.p)
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{C}, f::IndPoint, x::AbstractArray{C}, gamma=R(1)) where {R <: Real, C <: Union{R, Complex{R}}}
    y .= f.p
    return R(0)
end

function prox_naive(f::IndPoint, x::AbstractArray{C}, gamma=R(1)) where {R <: Real, C <: Union{R, Complex{R}}}
    y = similar(x)
    y .= f.p
    return y, R(0)
end
