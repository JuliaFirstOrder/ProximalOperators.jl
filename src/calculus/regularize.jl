# Regularize

export Regularize

"""
    Regularize(f, ρ=1.0, a=0.0)

Given function `f`, and optional parameters `ρ` (positive) and `a`, return
```math
g(x) = f(x) + \\tfrac{ρ}{2}\\|x-a\\|².
```
Parameter `a` can be either an array or a scalar, in which case it is subtracted component-wise from `x` in the above expression.
"""
struct Regularize{T, S, A}
    f::T
    rho::S
    a::A
    function Regularize{T,S,A}(f::T, rho::S, a::A) where {T, S, A}
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

Regularize(f::T, rho::S=1, a::A=0) where {T, S, A} = Regularize{T, S, A}(f, rho, a)

function (g::Regularize)(x)
    return g.f(x) + g.rho/2*norm(x .- g.a)^2
end

function gradient!(y, g::Regularize, x)
    v = gradient!(y, g.f, x)
    y .+= g.rho * (x .- g.a)
    return v + g.rho / 2 * norm(x .- g.a)^2
end

function prox!(y, g::Regularize, x, gamma)
    R = real(eltype(x))
    gr = g.rho * gamma
    gr2 = R(1) ./ (R(1) .+ gr)
    v = prox!(y, g.f, gr2 .* (x .+ gr .* g.a), gr2 .* gamma)
    return v + g.rho / R(2) * norm(y .- g.a)^2
end

function prox_naive(g::Regularize, x, gamma)
    R = real(eltype(x))
    y, v = prox_naive(g.f, x./(R(1) .+ gamma.*g.rho) .+ g.a./(R(1)./(gamma.*g.rho) .+ R(1)), gamma./(R(1) .+ gamma.*g.rho))
    return y, v + g.rho/R(2)*norm(y .- g.a)^2
end
