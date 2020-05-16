### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a Cholesky factorization of A'A + I/(lambda*gamma)
# or AA' + I/(lambda*gamma), according to which matrix is smaller.
# The factorization is cached and recomputed whenever gamma changes

using LinearAlgebra
using SparseArrays
using SuiteSparse

mutable struct LeastSquaresDirect{R <: Real, C <: RealOrComplex{R}, M <: AbstractMatrix{C}, V <: AbstractArray{C}, F <: Factorization} <: LeastSquares
    A::M # m-by-n
    b::V # m (by-p)
    lambda::R
    lambdaAtb::V
    gamma::Union{Nothing, R}
    shape::Symbol
    S::M
    res::Array{C} # m (by-p)
    q::Array{C} # n (by-p)
    fact::F
    function LeastSquaresDirect{R, C, M, V, F}(A::M, b::V, lambda::R) where {R <: Real, C <: RealOrComplex{R}, M <: AbstractMatrix{C}, V <: AbstractArray{C}, F <: Factorization}
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
        new(A, b, lambda, lambda*(A'*b), nothing, shape, S, zero(b), zeros(C, x_shape))
    end
end

is_convex(f::LeastSquaresDirect) = f.lambda >= 0
is_concave(f::LeastSquaresDirect) = f.lambda <= 0

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, C <: Union{R, Complex{R}}, M <: DenseMatrix{C}, V <: AbstractArray{C}}
    LeastSquaresDirect{R, C, M, V, Cholesky{C, M}}(A, b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, C <: Union{R, Complex{R}}, I <: Integer, M <: SparseMatrixCSC{C, I}, V <: AbstractArray{C}}
    LeastSquaresDirect{R, C, M, V, SuiteSparse.CHOLMOD.Factor{C}}(A, b, lambda)
end

# Adjoint/Transpose versions
function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, C <: Union{R, Complex{R}}, M <: TransposeOrAdjoint{<:DenseMatrix{C}}, V <: AbstractArray{C}}
    LeastSquaresDirect(copy(A), b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, C <: Union{R, Complex{R}}, I <: Integer, M <: TransposeOrAdjoint{<:SparseMatrixCSC{C, I}}, V <: AbstractArray{C}}
    LeastSquaresDirect(copy(A), b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, C <: Union{R, Complex{R}}, M <: AbstractMatrix{C}, V <: AbstractArray{C}}
    @warn "Could not infer type of Factorization for $M in LeastSquaresDirect, this type will be type-unstable"
    LeastSquaresDirect{R, C, M, V, Factorization}(A, b, lambda)
end

function (f::LeastSquaresDirect)(x::AbstractArray)
    mul!(f.res, f.A, x)
    f.res .-= f.b
    return (f.lambda / 2) * norm(f.res, 2)^2
end

function prox!(y::AbstractArray{C}, f::LeastSquaresDirect{R, C, M, V, F}, x::AbstractArray{C}, gamma::R=R(1)) where {R, C, M, V, F}
    # if gamma different from f.gamma then call factor_step!
    if gamma != f.gamma
        factor_step!(f, gamma)
    end
    solve_step!(y, f, x, gamma)
    mul!(f.res, f.A, y)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function factor_step!(f::LeastSquaresDirect{R, C, M, V, F}, gamma::R) where {R, C, M, V, F}
    f.fact = cholesky(f.S + I/gamma)
    f.gamma = gamma
end

function factor_step!(f::LeastSquaresDirect{R, C, M, V, F}, gamma::R) where {R, C, M <: SparseMatrixCSC, V, F}
    f.fact = ldlt(f.S; shift = R(1)/gamma)
    f.gamma = gamma
end

function solve_step!(y::AbstractArray{C}, f::LeastSquaresDirect{R, C, M, V, F}, x::AbstractArray{C}, gamma::R) where {R, C, M, V, F <: Cholesky{C, M}}
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

function solve_step!(y::AbstractArray{C}, f::LeastSquaresDirect{R, C, M, V, F}, x::AbstractArray{C}, gamma::R) where {R, C, M, V, F}
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

function gradient!(y::AbstractArray{C}, f::LeastSquaresDirect{R, C, M, V, F}, x::AbstractArray{C}) where {R, C, M, V, F}
    mul!(f.res, f.A, x)
    f.res .-= f.b
    mul!(y, adjoint(f.A), f.res)
    y .*= f.lambda
    fy = (f.lambda/2)*dot(f.res, f.res)
end

function prox_naive(f::LeastSquaresDirect{R, C}, x::AbstractArray{C}, gamma::R=R(1)) where {R, C <: RealOrComplex{R}}
    lamgam = f.lambda*gamma
    y = (f.A'*f.A + I/lamgam)\(f.A' * f.b + x/lamgam)
    fy = (f.lambda/2)*norm(f.A*y-f.b)^2
    return y, fy
end
