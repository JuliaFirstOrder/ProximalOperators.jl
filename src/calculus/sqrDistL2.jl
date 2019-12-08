# squared Euclidean distance from a set

export SqrDistL2

"""
**Squared distance from a convex set**

    SqrDistL2(ind_S, λ=1.0)

Given `ind_S` the indicator function of a convex set ``S``, and an optional positive parameter `λ`, returns the (weighted) squared Euclidean distance from ``S``, that is function
```math
g(x) = \\tfrac{λ}{2}\\mathrm{dist}_S^2(x) = \\min \\left\\{ \\tfrac{λ}{2}\\|y - x\\|^2 : y \\in S \\right\\}.
```
"""
struct SqrDistL2{R <: Real, T <: ProximableFunction} <: ProximableFunction
    ind::T
    lambda::R
    function SqrDistL2{R,T}(ind::T, lambda::R) where {R <: Real, T<:ProximableFunction}
        if !is_convex(ind) || !is_set(ind)
            error("`ind` must be the indicator of a convex set")
        end
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(ind, lambda)
        end
    end
end

is_prox_accurate(f::SqrDistL2) = is_prox_accurate(f.ind)
is_convex(f::SqrDistL2) = true
is_smooth(f::SqrDistL2) = true
is_quadratic(f::SqrDistL2) = is_affine(f.ind)
is_strongly_convex(f::SqrDistL2) = is_singleton(f.ind)

SqrDistL2(ind::T, lambda=1e0) where {T <: ProximableFunction} = SqrDistL2{typeof(lambda), T}(ind, lambda)

function (f::SqrDistL2)(x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    p, = prox(f.ind, x)
    return R(f.lambda/2) * normdiff2(x, p)
end

function prox!(y::AbstractArray{T}, f::SqrDistL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    p, = prox(f.ind, x)
    sqrd = R(f.lambda/2) * normdiff2(x, p)
    c1 = 1/(1 + R(f.lambda) * gamma)
    c2 = R(f.lambda) * gamma * c1
    for k in eachindex(p)
        y[k] = c1 * x[k] + c2 * p[k]
    end
    return sqrd * c1^2
end

function gradient!(y::AbstractArray{T}, f::SqrDistL2, x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    p, = prox(f.ind, x)
    dist2 = normdiff2(x,p)
    y .= R(f.lambda) .* (x .- p)
    return R(f.lambda/2) * dist2
end

fun_name(f::SqrDistL2) = "squared Euclidean distance from a convex set"
fun_dom(f::SqrDistL2) = fun_dom(f.ind)
fun_expr(f::SqrDistL2) = "x ↦ (λ/2) inf { ||x-y||^2 : y ∈ S} "
fun_params(f::SqrDistL2) = string("λ = $(f.lambda), S = ", typeof(f.ind))

function prox_naive(f::SqrDistL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    p, = prox(f.ind, x)
    sqrd = R(f.lambda/2) * norm(x - p)^2
    gamlam = R(f.lambda) * gamma
    return 1/(1 + gamlam) * x + gamlam/(1 + gamlam) * p, sqrd/(1+gamlam)^2
end
