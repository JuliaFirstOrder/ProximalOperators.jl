# indicator of nonnegative orthant

export IndNonnegative

"""
    IndNonnegative()

Return the indicator of the nonnegative orthant
```math
C = \\{ x : x \\geq 0 \\}.
```
"""
struct IndNonnegative end

is_separable(f::Type{<:IndNonnegative}) = true
is_convex(f::Type{<:IndNonnegative}) = true
is_cone(f::Type{<:IndNonnegative}) = true

function (::IndNonnegative)(x)
    R = eltype(x)
    for k in eachindex(x)
        if x[k] < 0
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y, ::IndNonnegative, x, gamma)
    R = eltype(x)
    for k in eachindex(x)
        if x[k] < 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    return R(0)
end

function prox_naive(::IndNonnegative, x, gamma)
    R = eltype(x)
    y = max.(R(0), x)
    return y, R(0)
end
