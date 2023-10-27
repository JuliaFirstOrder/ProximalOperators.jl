### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a Cholesky factorization of A'A + I/(lambda*gamma)
# or AA' + I/(lambda*gamma), according to which matrix is smaller.
# The factorization is cached and recomputed whenever gamma changes

using LinearAlgebra
using SparseArrays

mutable struct LeastSquaresDirect{N, R, C, M, V, F, IsConvex} <: LeastSquares
    A::M # m-by-n
    b::V # m (by-p)
    lambda::R
    lambdaAtb::V
    gamma::R
    shape::Symbol
    S::M
    res::Array{C, N} # m (by-p)
    q::Array{C, N} # n (by-p)
    fact::F
    function LeastSquaresDirect{N, R, C, M, V, F, IsConvex}(A::M, b::V, lambda::R) where {N, R, C, M, V, F, IsConvex}
        if size(A, 1) != size(b, 1)
            error("A and b have incompatible dimensions")
        end
        m, n = size(A)
        if m >= n
            S = lambda * (A' * A)
            shape = :Tall
        else
            S = lambda * (A * A')
            shape = :Fat
        end
        x_shape = infer_shape_of_x(A, b)
        new(A, b, lambda, lambda*(A'*b), R(-1), shape, S, zero(b), zeros(C, x_shape))
    end
end

is_convex(::Type{LeastSquaresDirect{N, R, C, M, V, F, IsConvex}}) where {N, R, C, M, V, F, IsConvex} = IsConvex

function LeastSquaresDirect(A::M, b, lambda) where M <: DenseMatrix
    C = eltype(M)
    R = real(C)
    LeastSquaresDirect{ndims(b), R, C, M, typeof(b), Cholesky{C, M}, lambda >= 0}(A, b, R(lambda))
end

function LeastSquaresDirect(A::M, b, lambda) where M <: SparseMatrixCSC
    C = eltype(M)
    R = real(C)
    LeastSquaresDirect{ndims(b), R, C, M, typeof(b), SparseArrays.CHOLMOD.Factor{C}, lambda >= 0}(A, b, R(lambda))
end

function LeastSquaresDirect(A::Union{Transpose, Adjoint}, b, lambda)
    LeastSquaresDirect(copy(A), b, lambda)
end

function LeastSquaresDirect(A, b, lambda)
    @warn "Could not infer type of Factorization for $M in LeastSquaresDirect, this type will be type-unstable"
    LeastSquaresDirect{N, R, C, M, V, Factorization, lambda >= 0}(A, b, lambda)
end

function (f::LeastSquaresDirect)(x)
    mul!(f.res, f.A, x)
    f.res .-= f.b
    return (f.lambda / 2) * norm(f.res, 2)^2
end

function prox!(y, f::LeastSquaresDirect, x, gamma)
    # if gamma different from f.gamma then call factor_step!
    if gamma != f.gamma
        factor_step!(f, gamma)
    end
    solve_step!(y, f, x, gamma)
    mul!(f.res, f.A, y)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function factor_step!(f::LeastSquaresDirect{N, R, C, M, V, F}, gamma) where {N, R, C, M, V, F}
    f.fact = cholesky(f.S + I/gamma)
    f.gamma = gamma
end

function factor_step!(f::LeastSquaresDirect{N, R, C, <:SparseMatrixCSC, V, F}, gamma) where {N, R, C, V, F}
    f.fact = ldlt(f.S; shift = R(1)/gamma)
    f.gamma = gamma
end

function solve_step!(y, f::LeastSquaresDirect{N, R, C, M, V, <:Cholesky}, x, gamma) where {N, R, C, M, V}
    f.q .= f.lambdaAtb .+ x./gamma
    # two cases: (1) tall A, (2) fat A
    if f.shape == :Tall
        # y .= f.fact\f.q
        y .= f.q
        LAPACK.trtrs!('U', 'C', 'N', f.fact.factors, y)
        LAPACK.trtrs!('U', 'N', 'N', f.fact.factors, y)
    else # f.shape == :Fat
        # y .= gamma*(f.q - lambda*(f.A'*(f.fact\(f.A*f.q))))
        mul!(f.res, f.A, f.q)
        LAPACK.trtrs!('U', 'C', 'N', f.fact.factors, f.res)
        LAPACK.trtrs!('U', 'N', 'N', f.fact.factors, f.res)
        mul!(y, adjoint(f.A), f.res)
        y .*= -f.lambda
        y .+= f.q
        y .*= gamma
    end
end

function solve_step!(y, f::LeastSquaresDirect, x, gamma)
    f.q .= f.lambdaAtb .+ x./gamma
    # two cases: (1) tall A, (2) fat A
    if f.shape == :Tall
        y .= f.fact\f.q
    else # f.shape == :Fat
        # y .= gamma*(f.q - lambda*(f.A'*(f.fact\(f.A*f.q))))
        mul!(f.res, f.A, f.q)
        f.res .= f.fact\f.res
        mul!(y, adjoint(f.A), f.res)
        y .*= -f.lambda
        y .+= f.q
        y .*= gamma
    end
end

function gradient!(y, f::LeastSquaresDirect, x)
    mul!(f.res, f.A, x)
    f.res .-= f.b
    mul!(y, adjoint(f.A), f.res)
    y .*= f.lambda
    return (f.lambda / 2) * real(dot(f.res, f.res))
end

function prox_naive(f::LeastSquaresDirect, x, gamma)
    lamgam = f.lambda*gamma
    y = (f.A'*f.A + I/lamgam)\(f.A' * f.b + x/lamgam)
    fy = (f.lambda/2)*norm(f.A*y-f.b)^2
    return y, fy
end
