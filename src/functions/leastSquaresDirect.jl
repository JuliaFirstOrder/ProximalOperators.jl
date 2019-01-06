### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a Cholesky factorization of A'A + I/(lambda*gamma)
# or AA' + I/(lambda*gamma), according to which matrix is smaller.
# The factorization is cached and recomputed whenever gamma changes

using LinearAlgebra
using SparseArrays
using SuiteSparse

mutable struct LeastSquaresDirect{R <: Real, RC <: RealOrComplex{R}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization} <: LeastSquares
	A::M # m-by-n matrix
	b::V
	Atb::V
	lambda::R
	gamma::R
	shape::Symbol
	S::M
	res::Vector{RC} # m-sized buffer
	q::Vector{RC} # n-sized buffer
	fact::F
	function LeastSquaresDirect{R, RC, M, V, F}(A::M, b::V, lambda::R) where {R <: Real, RC <: RealOrComplex{R}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization}
		if size(A, 1) != length(b)
			error("A and b have incompatible dimensions")
		end
		if lambda <= 0
			error("lambda must be positive")
		end
		m, n = size(A)
		if m >= n
			S = A'*A
			shape = :Tall
		else
			S = A*A'
			shape = :Fat
		end
		new(A, b, A'*b, lambda, -1, shape, S, zeros(RC, m), zeros(RC, n))
	end
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, M <: DenseMatrix{RC}, V <: AbstractVector{RC}}
	LeastSquaresDirect{R, RC, M, V, Cholesky{RC, M}}(A, b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, I <: Integer, M <: SparseMatrixCSC{RC, I}, V <: AbstractVector{RC}}
	LeastSquaresDirect{R, RC, M, V, SuiteSparse.CHOLMOD.Factor{RC}}(A, b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}}
	warn("Could not infer type of Factorization for $M in LeastSquaresDirect, this type will be type-unstable")
	LeastSquaresDirect{R, RC, M, V, Factorization}(A, b, lambda)
end

function (f::LeastSquaresDirect{R, RC, M, V, F})(x::AbstractVector{RC}) where {R, RC, M, V, F}
	mul!(f.res, f.A, x)
	f.res .-= f.b
	return (f.lambda/2)*norm(f.res, 2)^2
end

function prox!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}, gamma::R=R(1)) where {R, RC, M, V, F, D <: RealOrComplex{R}}
	# if gamma different from f.gamma then call factor_step!
	if gamma != f.gamma
		factor_step!(f, gamma)
	end
	solve_step!(y, f, x, gamma)
	mul!(f.res, f.A, y)
	f.res .-= f.b
	return (f.lambda/2)*norm(f.res, 2)^2
end

function factor_step!(f::LeastSquaresDirect{R, RC, M, V, F}, gamma::R) where {R, RC, M <: DenseMatrix, V, F}
	lamgam = f.lambda*gamma
	f.fact = cholesky(f.S + I/lamgam)
	f.gamma = gamma
end

function factor_step!(f::LeastSquaresDirect{R, RC, M, V, F}, gamma::R) where {R, RC, M <: SparseMatrixCSC, V, F}
	lamgam = f.lambda*gamma
	f.fact = ldlt(f.S; shift = 1.0/lamgam)
	f.gamma = gamma
end

function solve_step!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}, gamma::R) where {R, RC, M, V, F <: Cholesky{RC, M}, D <: RealOrComplex{R}}
	lamgam = f.lambda*gamma
	f.q .= f.Atb .+ x./lamgam
	# two cases: (1) tall A, (2) fat A
	if f.shape == :Tall
		# y .= f.fact\f.q
		y .= f.q
		LAPACK.trtrs!('U', 'C', 'N', f.fact.factors, y)
		LAPACK.trtrs!('U', 'N', 'N', f.fact.factors, y)
	else # f.shape == :Fat
		# y .= lamgam*(f.q - (f.A'*(f.fact\(f.A*f.q))))
		mul!(f.res, f.A, f.q)
		LAPACK.trtrs!('U', 'C', 'N', f.fact.factors, f.res)
		LAPACK.trtrs!('U', 'N', 'N', f.fact.factors, f.res)
		mul!(y, adjoint(f.A), f.res)
		y .-= f.q
		y .*= -lamgam
	end
end

function solve_step!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}, gamma::R) where {R, RC, M, V, F <: SuiteSparse.CHOLMOD.Factor{RC}, D <: RealOrComplex{R}}
	lamgam = f.lambda*gamma
	f.q .= f.Atb .+ x./lamgam
	# two cases: (1) tall A, (2) fat A
	if f.shape == :Tall
		y .= f.fact\f.q
	else # f.shape == :Fat
		# y .= lamgam*(f.q - (f.A'*(f.fact\(f.A*f.q))))
		mul!(f.res, f.A, f.q)
		f.res .= f.fact\f.res
		mul!(y, adjoint(f.A), f.res)
		y .-= f.q
		y .*= -lamgam
	end
end

function gradient!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}) where {R, RC, M, V, F, D <: Union{R, Complex{R}}}
	mul!(f.res, f.A, x)
	f.res .-= f.b
	mul!(y, adjoint(f.A), f.res)
	y .*= f.lambda
	fy = (f.lambda/2)*dot(f.res, f.res)
end

function prox_naive(f::LeastSquaresDirect, x::AbstractVector{D}, gamma::R=R(1)) where {R, D <: RealOrComplex{R}}
	lamgam = f.lambda*gamma
	y = (f.A'*f.A + I/lamgam)\(f.Atb + x/lamgam)
	fy = (f.lambda/2)*norm(f.A*y-f.b)^2
	return y, fy
end
