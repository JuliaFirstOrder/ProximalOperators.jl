# indicator of the free cone

export IndFree

"""
    IndFree()

Return the indicator function of the whole space, or "free cone", *i.e.*,
a function which is identically zero.
"""
struct IndFree end

is_separable(f::Type{<:IndFree}) = true
is_convex(f::Type{<:IndFree}) = true
is_affine_indicator(f::Type{<:IndFree}) = true
is_cone_indicator(f::Type{<:IndFree}) = true
is_smooth(f::Type{<:IndFree}) = true
is_generalized_quadratic(f::Type{<:IndFree}) = true

const Zero = IndFree

function (::IndFree)(x)
    return real(eltype(x))(0)
end

function prox!(y, ::IndFree, x, gamma)
    y .= x
    return real(eltype(x))(0)
end

function gradient!(y, ::IndFree, x)
    T = eltype(x)
    y .= T(0)
    return real(T)(0)
end

function prox_naive(::IndFree, x, gamma)
    return x, real(eltype(x))(0)
end
