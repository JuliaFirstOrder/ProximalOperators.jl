### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a Cholesky factorization of Q + I/gamma.
# The factorization is cached and recomputed whenever gamma changes

using LinearAlgebra
using SparseArrays
using SuiteSparse

mutable struct QuadraticDirect{R <: Real, M <: AbstractMatrix{R}, V <: AbstractVector{R}, F <: Factorization} <: Quadratic
    Q::M
    q::V
    gamma::R
    temp::V
    fact::F
    function QuadraticDirect{R, M, V, F}(Q::M, q::V) where {R <: Real, M <: AbstractMatrix{R}, V <: AbstractVector{R}, F <: Factorization}
        if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
            error("Q must be squared and q must be compatible with Q")
        end
        new(Q, q, -1, similar(q))
    end
end

function QuadraticDirect(Q::M, q::V) where {R <: Real, I <: Integer, M <: SparseMatrixCSC{R, I}, V <: AbstractVector{R}}
    QuadraticDirect{R, M, V, SuiteSparse.CHOLMOD.Factor{R}}(Q, q)
end

function QuadraticDirect(Q::M, q::V) where {R <: Real, M <: DenseMatrix{R}, V <: AbstractVector{R}}
    QuadraticDirect{R, M, V, Cholesky{R, M}}(Q, q)
end

function (f::QuadraticDirect{R, M, V, F})(x::AbstractArray{R}) where {R, M, V, F}
    mul!(f.temp, f.Q, x)
    return 0.5*dot(x, f.temp) + dot(x, f.q)
end

function prox!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}, gamma::R=R(1)) where {R, M, V, F <: Cholesky}
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

function prox!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}, gamma::R=R(1)) where {R, M, V, F <: SuiteSparse.CHOLMOD.Factor}
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

function factor_step!(f::QuadraticDirect{R, M, V, F}, gamma::R) where {R, M <: DenseMatrix{R}, V, F}
    f.gamma = gamma
    f.fact = cholesky(f.Q + I/gamma)
end

function factor_step!(f::QuadraticDirect{R, M, V, F}, gamma::R) where {R, I, M <: SparseMatrixCSC{R, I}, V, F}
    f.gamma = gamma
    f.fact = ldlt(f.Q; shift = 1/gamma)
end

function gradient!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}) where {R, M, V, F}
    mul!(y, f.Q, x)
    y .+= f.q
    return 0.5*(dot(x, y) + dot(x, f.q))
end

function prox_naive(f::QuadraticDirect, x::AbstractArray{R}, gamma::R=one(R)) where R
    y = (gamma*f.Q + I)\(x - gamma*f.q)
    fy = 0.5*dot(y, f.Q*y) + dot(y, f.q)
    return y, fy
end
