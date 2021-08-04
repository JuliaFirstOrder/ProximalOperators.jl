# indicator of the free cone

export IndFree

"""
**Indicator of the free cone**

    IndFree()

Returns the indicator function of the whole space, or "free cone", *i.e.*,
a function which is identically zero.
"""
struct IndFree end

is_separable(f::IndFree) = true
is_convex(f::IndFree) = true
is_affine(f::IndFree) = true
is_cone(f::IndFree) = true
is_smooth(f::IndFree) = true
is_quadratic(f::IndFree) = true

const Zero = IndFree

function (f::IndFree)(x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
    return R(0)
end

function prox!(y::AbstractArray{T}, f::IndFree, x::AbstractArray{T}, gamma) where {R, T <: RealOrComplex{R}}
    y .= x
    return R(0)
end

function gradient!(y::AbstractArray{T}, f::IndFree, x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
    y .= T(0)
    return R(0)
end

function prox_naive(f::IndFree, x::AbstractArray{R}, gamma) where {R}
    return x, R(0)
end
