# indicator of nonnegative orthant

export IndNonnegative

"""
**Indicator of the nonnegative orthant**

    IndNonnegative()

Returns the indicator of the set
```math
C = \\{ x : x \\geq 0 \\}.
```
"""
struct IndNonnegative end

is_separable(f::Type{<:IndNonnegative}) = true
is_convex(f::Type{<:IndNonnegative}) = true
is_cone(f::Type{<:IndNonnegative}) = true

function (f::IndNonnegative)(x::AbstractArray{R}) where R <: Real
    for k in eachindex(x)
        if x[k] < 0
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y::AbstractArray{R}, f::IndNonnegative, x::AbstractArray{R}, gamma) where R <: Real
    for k in eachindex(x)
        if x[k] < 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    return R(0)
end

function prox_naive(f::IndNonnegative, x::AbstractArray{R}, gamma) where R <: Real
    y = max.(R(0), x)
    return y, R(0)
end
