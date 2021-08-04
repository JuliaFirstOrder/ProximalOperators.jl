# indicator of nonpositive orthant

export IndNonpositive

"""
**Indicator of the nonpositive orthant**

    IndNonpositive()

Returns the indicator of the set
```math
C = \\{ x : x \\leq 0 \\}.
```
"""
struct IndNonpositive end

is_separable(f::IndNonpositive) = true
is_convex(f::IndNonpositive) = true
is_cone(f::IndNonpositive) = true

function (f::IndNonpositive)(x::AbstractArray{R}) where R <: Real
    for k in eachindex(x)
        if x[k] > 0
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y::AbstractArray{R}, f::IndNonpositive, x::AbstractArray{R}, gamma=R(1)) where R <: Real
    for k in eachindex(x)
        if x[k] > 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    return R(0)
end

function prox_naive(f::IndNonpositive, x::AbstractArray{R}, gamma=R(1)) where R <: Real
    y = min.(R(0), x)
    return y, R(0)
end
