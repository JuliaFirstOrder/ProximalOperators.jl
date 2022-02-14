# indicator of nonpositive orthant

export IndNonpositive

"""
    IndNonpositive()

Return the indicator of the nonpositive orthant
```math
C = \\{ x : x \\leq 0 \\}.
```
"""
struct IndNonpositive end

is_separable(f::Type{<:IndNonpositive}) = true
is_convex(f::Type{<:IndNonpositive}) = true
is_cone(f::Type{<:IndNonpositive}) = true

function (::IndNonpositive)(x)
    R = eltype(x)
    for k in eachindex(x)
        if x[k] > 0
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y, ::IndNonpositive, x, gamma)
    R = eltype(x)
    for k in eachindex(x)
        if x[k] > 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    return R(0)
end

function prox_naive(::IndNonpositive, x, gamma)
    R = eltype(x)
    y = min.(R(0), x)
    return y, R(0)
end
