using LinearAlgebra

mutable struct EpicomposeQuadratic{F, M, P, V, R} <: Epicompose
    L::M
    Q::P
    q::V
    gamma::Maybe{R}
    fact::Maybe{F}
    function EpicomposeQuadratic{F, M, P, V, R}(L::M, Q::P, q::V) where {
        R <: Real,
        M <: AbstractMatrix{R},
        P <: AbstractMatrix{R},
        V <: AbstractVector{R},
        F <: Factorization,
    }
        new(L, Q, q, nothing, nothing)
    end
end

function EpicomposeQuadratic(L, Q::P, q) where {R, P <: DenseMatrix{R}}
    return EpicomposeQuadratic{
        LinearAlgebra.Cholesky{R}, typeof(L), P, typeof(q), real(eltype(L))
    }(L, Q, q)
end

function EpicomposeQuadratic(L, Q::P, q) where {R, P <: SparseMatrixCSC{R}}
    return EpicomposeQuadratic{
        SuiteSparse.CHOLMOD.Factor{R}, typeof(L), P, typeof(q), real(eltype(L))
    }(L, Q, q)
end

# TODO: enable construction from other types of quadratics, e.g. LeastSquares
# TODO: probably some access methods are needed to obtain Hessian and linear term?

function EpicomposeQuadratic(L, f::Q) where {Q <: Quadratic}
    return EpicomposeQuadratic(L, f.Q, f.q)
end

function factor_step!(g::EpicomposeQuadratic, gamma)
    g.gamma = gamma
    g.fact = cholesky(g.Q + (g.L' * g.L)/gamma)
end

function prox!(y, g::EpicomposeQuadratic, x, gamma)
    if g.gamma === nothing || !isapprox(gamma, g.gamma)
        factor_step!(g, gamma)
    end
    p = g.fact\((g.L' * x) / gamma - g.q)
    fy = dot(p, g.Q * p)/2 + dot(p, g.q)
    mul!(y, g.L, p)
    return fy
end

function prox_naive(g::EpicomposeQuadratic, x, gamma)
    S = g.Q + (g.L' * g.L) / gamma
    p = S\((g.L' * x) / gamma - g.q)
    fy = dot(p, g.Q * p)/2 + dot(p, g.q)
    y = g.L * p
    return y, fy
end
