# indicator of the zero cone

export IndZero

"""
    IndZero()

Return the indicator function of the set containing the origin, the "zero cone".
"""
struct IndZero end

is_separable(f::Type{<:IndZero}) = true
is_convex(f::Type{<:IndZero}) = true
is_singleton(f::Type{<:IndZero}) = true
is_cone(f::Type{<:IndZero}) = true
is_affine(f::Type{<:IndZero}) = true

function (::IndZero)(x)
    C = eltype(x)
    for k in eachindex(x)
        if x[k] != C(0)
            return real(C)(Inf)
        end
    end
    return real(C)(0)
end

function prox!(y, ::IndZero, x, gamma)
    for k in eachindex(y)
        y[k] = eltype(y)(0)
    end
    return real(eltype(x))(0)
end

function prox_naive(::IndZero, x, gamma)
    return zero(x), real(eltype(x))(0)
end
