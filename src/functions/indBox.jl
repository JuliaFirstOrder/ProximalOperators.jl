# indicator of a generic box

export IndBox, IndBallLinf

"""
    IndBox(low, up)

Return the indicator function of the box
```math
S = \\{ x : low \\leq x \\leq up \\}.
```
Parameters `low` and `up` can be either scalars or arrays of the same dimension as the space: they must satisfy `low <= up`, and are allowed to take values `-Inf` and `+Inf` to indicate unbounded coordinates.
"""
struct IndBox{T, S}
    lb::T
    ub::S
    function IndBox{T,S}(lb::T, ub::S) where {T, S}
        if !(eltype(lb) <: Real && eltype(ub) <: Real)
            error("`lb` and `ub` must be real")
        end
        if any(lb .> ub)
            error("`lb` and `ub` must satisfy `lb <= ub`")
        else
            new(lb, ub)
        end
    end
end

is_separable(f::Type{<:IndBox}) = true
is_convex(f::Type{<:IndBox}) = true
is_set_indicator(f::Type{<:IndBox}) = true

compatible_bounds(::Real, ::Real) = true
compatible_bounds(::Real, ::AbstractArray) = true
compatible_bounds(::AbstractArray, ::Real) = true
compatible_bounds(lb::AbstractArray, ub::AbstractArray) = size(lb) == size(ub)

IndBox(lb, ub) = if compatible_bounds(lb, ub)
    IndBox{typeof(lb), typeof(ub)}(lb, ub)
else
    error("bounds must have the same dimensions, or at least one of them be scalar")
end

function (f::IndBox)(x)
    R = eltype(x)
    for k in eachindex(x)
        if x[k] < get_kth_elem(f.lb, k) || x[k] > get_kth_elem(f.ub, k)
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y, f::IndBox, x, gamma)
    for k in eachindex(x)
        if x[k] < get_kth_elem(f.lb, k)
            y[k] = get_kth_elem(f.lb, k)
        elseif x[k] > get_kth_elem(f.ub, k)
            y[k] = get_kth_elem(f.ub, k)
        else
            y[k] = x[k]
        end
    end
    return eltype(x)(0)
end

"""
**Indicator of a ``L_∞`` norm ball**

    IndBallLinf(r=1.0)

Return the indicator function of the set
```math
S = \\{ x : \\max (|x_i|) \\leq r \\}.
```
Parameter `r` must be positive.
"""
IndBallLinf(r::R=1) where R = IndBox(-r, r)

function prox_naive(f::IndBox, x, gamma)
    y = min.(f.ub, max.(f.lb, x))
    return y, eltype(x)(0)
end
