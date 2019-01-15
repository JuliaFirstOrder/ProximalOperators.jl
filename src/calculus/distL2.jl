# Euclidean distance from a set

export DistL2

"""
**Distance from a convex set**

    DistL2(ind_S)

Given `ind_S` the indicator function of a convex set ``S``, and an optional positive parameter `λ`, returns the (weighted) Euclidean distance from ``S``, that is function
```math
g(x) = λ\\mathrm{dist}_S(x) = \\min \\{ λ\\|y - x\\| : y \\in S \\}.
```
"""
struct DistL2{R <: Real, T <: ProximableFunction} <: ProximableFunction
    ind::T
    lambda::R
    function DistL2{R, T}(ind::T, lambda::R) where {R <: Real, T <: ProximableFunction}
        if !is_set(ind) || !is_convex(ind)
            error("`ind` must be a convex set")
        end
        if lambda < 0
            error("parameter `λ` must be nonnegative")
        else
            new(ind, lambda)
        end
    end
end

is_prox_accurate(f::DistL2) = is_prox_accurate(f.ind)
is_convex(f::DistL2) = is_convex(f.ind)

DistL2(ind::T, lambda::R=1.0) where {R <: Real, T <: ProximableFunction} = DistL2{R, T}(ind, lambda)

function (f::DistL2)(x::AbstractArray{R}) where R <: RealOrComplex
    p, = prox(f.ind, x)
    return f.lambda*normdiff(x,p)
end

function prox!(y::AbstractArray{T}, f::DistL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    prox!(y, f.ind, x)
    d = normdiff(x,y)
    gamlam = (gamma*f.lambda)
    if gamlam < d
        gamlamd = gamlam/d
        y .= (1-gamlamd).*x .+ gamlamd.*y
        return f.lambda*(d-gamlam)
    end
    return R(0)
end

function gradient!(y::AbstractArray{T}, f::DistL2, x::AbstractArray{T}) where T <: RealOrComplex
    prox!(y, f.ind, x) # Use y as temporary storage
    dist = normdiff(x,y)
    if dist > 0
        y .= (f.lambda/dist).*(x .- y)
    else
        y .= 0
    end
    return f.lambda*dist
end

fun_name(f::DistL2) = "Euclidean distance from a convex set"
fun_dom(f::DistL2) = fun_dom(f.ind)
fun_expr(f::DistL2) = "x ↦ λ inf { ||x-y|| : y ∈ S} "
fun_params(f::DistL2) = string("λ = $(f.lambda), S = ", typeof(f.ind))

function prox_naive(f::DistL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    p, = prox(f.ind, x)
    d = norm(x-p)
    gamlam = gamma*f.lambda
    if d > gamlam
        return x + gamlam/d*(p-x), f.lambda*(d-gamlam)
    end
    return p, R(0)
end
