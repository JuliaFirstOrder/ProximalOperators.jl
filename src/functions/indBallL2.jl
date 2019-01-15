# indicator of the L2 norm ball with given radius

export IndBallL2

"""
**Indicator of a Euclidean ball**

    IndBallL2(r=1.0)

Returns the indicator function of the set
```math
S = \\{ x : \\|x\\| \\leq r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.
"""
struct IndBallL2{R <: Real} <: ProximableFunction
    r::R
    function IndBallL2{R}(r::R) where {R <: Real}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::IndBallL2) = true
is_set(f::IndBallL2) = true

IndBallL2(r::R=1.0) where {R <: Real} = IndBallL2{R}(r)

function (f::IndBallL2)(x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    if norm(x) - f.r > f.r*eps(R)
        return R(Inf)
    end
    return R(0)
end

function prox!(y::AbstractArray{T}, f::IndBallL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
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

fun_name(f::IndBallL2) = "indicator of an L2 norm ball"
fun_dom(f::IndBallL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndBallL2) = "x ↦ 0 if ||x|| ⩽ r, +∞ otherwise"
fun_params(f::IndBallL2) = "r = $(f.r)"

function prox_naive(f::IndBallL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    normx = norm(x)
    if normx > f.r
        y = (f.r/normx)*x
    else
        y = x
    end
    return y, R(0)
end
