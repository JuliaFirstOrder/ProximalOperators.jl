### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a Cholesky factorization of Q + I/gamma.
# The factorization is cached and recomputed whenever gamma changes

using LinearAlgebra
using SparseArrays
using SuiteSparse

mutable struct QuadraticDirect{R, M, V, F} <: Quadratic
    Q::M
    q::V
    gamma::R
    temp::V
    fact::F
    function QuadraticDirect{R, M, V, F}(Q::M, q::V) where {R, M, V, F}
        if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
            error("Q must be squared and q must be compatible with Q")
        end
        new(Q, q, -1, similar(q))
    end
end

function QuadraticDirect(Q::M, q) where M <: SparseMatrixCSC
    R = eltype(M)
    QuadraticDirect{R, M, typeof(q), SuiteSparse.CHOLMOD.Factor{R}}(Q, q)
end

function QuadraticDirect(Q::M, q) where M <: DenseMatrix
    R = eltype(M)
    QuadraticDirect{R, M, typeof(q), Cholesky{R, M}}(Q, q)
end

function (f::QuadraticDirect)(x)
    mul!(f.temp, f.Q, x)
    return 0.5*dot(x, f.temp) + dot(x, f.q)
end

function prox!(y, f::QuadraticDirect{R, M, V, <:Cholesky}, x, gamma) where {R, M, V}
    if gamma != f.gamma
        factor_step!(f, gamma)
    end
    y .= x./gamma
    y .-= f.q
    # Qy = U'Uy = b, therefore y = U\(U'\b)
    LAPACK.trtrs!('U', 'C', 'N', f.fact.factors, y)
    LAPACK.trtrs!('U', 'N', 'N', f.fact.factors, y)
    mul!(f.temp, f.Q, y)
    fy = 0.5*dot(y, f.temp) + dot(y, f.q)
    return fy
end

function prox!(y, f::QuadraticDirect{R, M, V, <:SuiteSparse.CHOLMOD.Factor}, x, gamma) where {R, M, V}
    if gamma != f.gamma
        factor_step!(f, gamma)
    end
    f.temp .= x./gamma
    f.temp .-= f.q
    y .= f.fact\f.temp
    mul!(f.temp, f.Q, y)
    fy = 0.5*dot(y, f.temp) + dot(y, f.q)
    return fy
end

function factor_step!(f::QuadraticDirect{R, <:DenseMatrix, V, F}, gamma) where {R, V, F}
    f.gamma = gamma
    f.fact = cholesky(f.Q + I/gamma)
end

function factor_step!(f::QuadraticDirect{R, <:SparseMatrixCSC, V, F}, gamma) where {R, V, F}
    f.gamma = gamma
    f.fact = ldlt(f.Q; shift = 1/gamma)
end

function gradient!(y, f::QuadraticDirect{R, M, V, F}, x) where {R, M, V, F}
    mul!(y, f.Q, x)
    y .+= f.q
    return 0.5*(dot(x, y) + dot(x, f.q))
end

function prox_naive(f::QuadraticDirect, x, gamma)
    y = (gamma*f.Q + I)\(x - gamma*f.q)
    fy = 0.5*dot(y, f.Q*y) + dot(y, f.q)
    return y, fy
end
