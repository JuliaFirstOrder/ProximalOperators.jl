# indicator of the zero cone

export IndZero

"""
**Indicator of the zero cone**

    IndZero()

Returns the indicator function of the set containing the origin, the "zero cone".
"""
struct IndZero end

is_separable(f::IndZero) = true
is_convex(f::IndZero) = true
is_singleton(f::IndZero) = true
is_cone(f::IndZero) = true
is_affine(f::IndZero) = true

function (f::IndZero)(x::AbstractArray{C}) where {R <: Real, C <: Union{R, Complex{R}}}
    for k in eachindex(x)
        if x[k] != C(0)
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y::AbstractArray{C}, f::IndZero, x::AbstractArray{C}, gamma=R(1)) where {R <: Real, C <: Union{R, Complex{R}}}
    for k in eachindex(x)
        y[k] = C(0)
    end
    return R(0)
end

function prox_naive(f::IndZero, x::AbstractArray{C}, gamma=R(1)) where {R <: Real, C <: Union{R, Complex{R}}}
    return zero(x), R(0)
end
