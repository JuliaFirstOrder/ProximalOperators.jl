### CONCRETE TYPE: DIRECT PROX EVALUATION

mutable struct LeastSquaresDirect{R <: Real, RC <: RealOrComplex{R}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization} <: LeastSquares
  A::M
  b::V
  Atb::V
  lambda::R
  gamma::R
  S::M
  res::Vector{RC}
  U::F
  function LeastSquaresDirect{R, RC, M, V, F}(A::M, b::V, lambda::R) where {R <: Real, RC <: RealOrComplex{R}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization}
    if size(A, 1) != length(b)
      error("A and b have incompatible dimensions")
    end
    if lambda <= 0
      error("lambda must be positive")
    end
    if size(A,1) >= size(A,2)
      new(A, b, A'*b, lambda, -1, A'*A, zeros(RC, length(b)))
    else
      new(A, b, A'*b, lambda, -1, A*A', zeros(RC, length(b)))
    end
  end
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, M <: DenseMatrix{RC}, V <: AbstractVector{RC}}
  LeastSquaresDirect{R, RC, M, V, LinAlg.Cholesky{RC, M}}(A, b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, I <: Integer, M <: SparseMatrixCSC{RC, I}, V <: AbstractVector{RC}}
  LeastSquaresDirect{R, RC, M, V, SparseArrays.CHOLMOD.Factor{RC}}(A, b, lambda)
end

function LeastSquaresDirect(A::M, b::V, lambda::R) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}}
  warn("Could not infer type of Factorization for $M in LeastSquaresDirect, this type will be type-unstable")
  LeastSquaresDirect{R, RC, M, V, Factorization}(A, b, lambda)
end

function (f::LeastSquaresDirect{R, RC, M, V, F})(x::AbstractVector{RC}) where {R, RC, M, V, F}
  A_mul_B!(f.res, f.A, x)
  f.res .-= f.b
  return (f.lambda/2)*vecnorm(f.res, 2)^2
end

function factor_step!(f::LeastSquaresDirect{R, RC, M, V, F}, gamma::R) where {R, RC, M <: DenseMatrix, V, F}
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S + I/lamgam)
  f.gamma = gamma
end

function factor_step!(f::LeastSquaresDirect{R, RC, M, V, F}, gamma::R) where {R, RC, M <: SparseMatrixCSC, V, F}
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S; shift=1.0/lamgam)
  f.gamma = gamma
end

function prox!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}, gamma::R=one(R)) where {R, RC, M, V, F, D <: RealOrComplex{R}}
  # if gamma different from f.gamma then call factor_step!
  if gamma != f.gamma
    factor_step!(f, gamma)
  end
  lamgam = f.lambda*gamma
  # solve step, two cases: (1) tall A, (2) fat A
  q = f.Atb + x/lamgam
  if size(f.A,1) >= size(f.A,2)
    y .= f.U\q
  else
    y .= lamgam*(q - (f.A'*(f.U\(f.A*q))))
  end
  A_mul_B!(f.res, f.A, y)
  f.res .-= f.b
  return (f.lambda/2)*norm(f.res, 2)^2
end

function gradient!(y::AbstractVector{D}, f::LeastSquaresDirect{R, RC, M, V, F}, x::AbstractVector{D}) where {R, RC, M, V, F, D <: Union{R, Complex{R}}}
  A_mul_B!(f.res, f.A, x)
  f.res .-= f.b
  Ac_mul_B!(y, f.A, f.res)
  y .*= f.lambda
  fy = (f.lambda/2)*dot(f.res, f.res)
end

function prox_naive(f::LeastSquaresDirect, x::AbstractVector, gamma=1.0)
  lamgam = f.lambda*gamma
  y = (f.A'*f.A + I/lamgam)\(f.Atb + x/lamgam)
  fy = (f.lambda/2)*norm(f.A*y-f.b)^2
  return y, fy
end
