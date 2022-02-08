# indicator of the L2 norm ball with given radius

export IndBallL2

"""
    IndBallL2(r=1.0)

Return the indicator function of the Euclidean ball
```math
S = \\{ x : \\|x\\| \\leq r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.
"""
struct IndBallL2{R <: Real}
    r::R
    function IndBallL2{R}(r::R) where {R <: Real}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::Type{<:IndBallL2}) = true
is_set(f::Type{<:IndBallL2}) = true

IndBallL2(r::R=1.0) where {R <: Real} = IndBallL2{R}(r)

function (f::IndBallL2)(x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    if isapprox_le(norm(x), f.r, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{T}, f::IndBallL2, x::AbstractArray{T}, gamma) where {R <: Real, T <: RealOrComplex{R}}
    scal = f.r/norm(x)
    if scal > 1
        y .= x
        return R(0)
    end
    for k in eachindex(x)
        y[k] = scal*x[k]
    end
    return R(0)
end

function prox_naive(f::IndBallL2, x::AbstractArray{T}, gamma) where {R <: Real, T <: RealOrComplex{R}}
    normx = norm(x)
    if normx > f.r
        y = (f.r/normx)*x
    else
        y = x
    end
    return y, R(0)
end
