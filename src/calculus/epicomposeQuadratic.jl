using LinearAlgebra

mutable struct EpicomposeQuadratic{F, M, P, V, R} <: Epicompose
    L::M
    Q::P
    q::V
    gamma::R
    fact::F
    function EpicomposeQuadratic{F, M, P, V, R}(L::M, Q::P, q::V) where {F, M, P, V, R}
        new(L, Q, q, -R(1))
    end
end

function EpicomposeQuadratic(L, Q::P, q) where {R, P <: DenseMatrix{R}}
    return EpicomposeQuadratic{
        LinearAlgebra.Cholesky{R, P}, typeof(L), P, typeof(q), real(eltype(L))
    }(L, Q, q)
end

function EpicomposeQuadratic(L, Q::P, q) where {R, P <: SparseMatrixCSC{R}}
    return EpicomposeQuadratic{
        SuiteSparse.CHOLMOD.Factor{R}, typeof(L), P, typeof(q), real(eltype(L))
    }(L, Q, q)
end

# TODO: enable construction from other types of quadratics, e.g. LeastSquares
# TODO: probably some access methods are needed to obtain Hessian and linear term?

EpicomposeQuadratic(L, f::Quadratic) = EpicomposeQuadratic(L, f.Q, f.q)

function get_factorization!(g::EpicomposeQuadratic, gamma)
    if !isapprox(gamma, g.gamma)
        g.gamma = gamma
        g.fact = cholesky(g.Q + (g.L' * g.L)/gamma)
    end
    return g.fact
end

function prox!(y, g::EpicomposeQuadratic, x, gamma)
    fact = get_factorization!(g, gamma)
    p = fact\((g.L' * x) / gamma - g.q)
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
