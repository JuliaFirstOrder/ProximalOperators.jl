using LinearAlgebra

mutable struct EpicomposeGramDiagonal{M, P, R} <: Epicompose
    L::M
    f::P
    mu::R
    function EpicomposeGramDiagonal{M, P, R}(L::M, f::P, mu::R) where {
        R <: Real, C <: Union{R, Complex{R}}, M, P <: ProximableFunction
    }
        if mu <= 0
            error("mu must be positive")
        end
        new(L, f, mu)
    end
end

EpicomposeGramDiagonal(L, f, mu) = EpicomposeGramDiagonal{typeof(L), typeof(f), typeof(mu)}(L, f, mu)

function prox!(y::AbstractArray{C}, g::EpicomposeGramDiagonal, x::AbstractArray{C}, gamma::R=one(R)) where {
    R <: Real, C <: Union{R, Complex{R}}
}
    z = (g.L'*x)/g.mu
    p, v = prox(g.f, z, gamma/g.mu)
    mul!(y, g.L, p)
    return v
end

function prox_naive(g::EpicomposeGramDiagonal, x::AbstractArray{C}, gamma::R=one(R)) where {
    R <: Real, C <: Union{R, Complex{R}}
}
    z = (g.L'*x)/g.mu
    p, v = prox_naive(g.f, z, gamma/g.mu)
    y = g.L*p
    return y, v
end
