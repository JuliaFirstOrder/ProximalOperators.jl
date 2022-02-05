# Regularize

export Regularize

"""
**Regularize**

    Regularize(f, ρ=1.0, a=0.0)

Given function `f`, and optional parameters `ρ` (positive) and `a`, returns
```math
g(x) = f(x) + \\tfrac{ρ}{2}\\|x-a\\|².
```
Parameter `a` can be either an array or a scalar, in which case it is subtracted component-wise from `x` in the above expression.
"""
struct Regularize{T, S <: Real, A <: Union{Real, AbstractArray}}
    f::T
    rho::S
    a::A
    function Regularize{T,S,A}(f::T, rho::S, a::A) where {T, S <: Real, A <: Union{Real, AbstractArray}}
        if rho <= 0.0
            error("parameter `ρ` must be positive")
        else
            new(f, rho, a)
        end
    end
end

is_separable(::Type{<:Regularize{T}}) where T = is_separable(T)
is_prox_accurate(::Type{<:Regularize{T}}) where T = is_prox_accurate(T)
is_convex(::Type{<:Regularize{T}}) where T = is_convex(T)
is_smooth(::Type{<:Regularize{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Regularize{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Regularize}) = true

Regularize(f::T, rho::S, a::A) where {T, S <: Real, A <: AbstractArray} = Regularize{T, S, A}(f, rho, a)

Regularize(f::T, rho::S=one(S), a::S=zero(S)) where {T, S <: Real} = Regularize{T, S, S}(f, rho, a)

function (g::Regularize)(x::AbstractArray{T}) where T <: RealOrComplex
    return g.f(x) + g.rho/2*norm(x .- g.a)^2
end

function gradient!(y::AbstractArray{T}, g::Regularize, x::AbstractArray{T}) where T <: RealOrComplex
    v = gradient!(y, g.f, x)
    y .+= g.rho*(x .- g.a)
    return v + g.rho/2*norm(x .- g.a)^2
end

function prox!(y::AbstractArray{T}, g::Regularize, x::AbstractArray{T}, gamma) where {R <: Real, T <: RealOrComplex{R}}
    gr = g.rho*gamma
    gr2 = 1.0 ./ (1.0 .+ gr)
    v = prox!(y, g.f, gr2.*(x .+ gr.*g.a), gr2.*gamma)
    return v + g.rho/2*norm(y .- g.a)^2
end

fun_name(f::Regularize) = string("Regularized ", fun_name(f.f))
fun_dom(f::Regularize) = fun_dom(f.f)
fun_expr(f::Regularize) = string(fun_expr(f.f), "+(ρ/2)||x-a||²")
fun_params(f::Regularize) = "ρ = $(f.rho), a = $( typeof(f.a) <: Real ? f.a : typeof(f.a) )"

function prox_naive(g::Regularize, x::AbstractArray{T}, gamma) where {R <: Real, T <: RealOrComplex{R}}
    y, v = prox_naive(g.f, x./(1.0 .+ gamma.*g.rho) .+ g.a./(1.0./(gamma.*g.rho) .+ 1.0), gamma./(1.0 .+ gamma.*g.rho))
    return y, v + g.rho/2*norm(y .- g.a)^2
end
