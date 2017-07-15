# least squares penalty

export LeastSquares

"""
**Least squares penalty**

    LeastSquares(A, b, λ=1.0)

For a matrix `A`, a vector `b` and a scalar `λ`, returns the function
```math
f(x) = \\tfrac{\\lambda}{2}\\|Ax - b\\|^2.
```
"""

type LeastSquares{R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractArray{RC, 2}, V <: AbstractArray{RC, 1}, F <: Factorization} <: ProximableFunction
  A::M
  b::V
  Atb::V
  lambda::R
  gamma::R
  S::M
  U::F
  function LeastSquares{R, RC, M, V, F}(A::M, b::V, lambda::R) where {RC <: RealOrComplex, R<:Real, M<:AbstractArray{RC,2}, V<:AbstractArray{RC,1}, F<:Factorization}
    if size(A, 1) != length(b)
      error("A and b have incompatible dimensions")
    end
    if lambda <= 0
      error("lambda must be positive")
    end
    if size(A,1) >= size(A,2)
      new(A, b, A'*b, lambda, -1, A'*A)
    else
      new(A, b, A'*b, lambda, -1, A*A')
    end
  end
end

function LeastSquares(A::M, b::V, lambda::R=one(R)) where {R <: Real, RC <: Union{R, Complex{R}}, I <: Integer, M <: SparseMatrixCSC{RC, I}, V <: AbstractArray{RC, 1}}
  LeastSquares{R, RC, M, V, SparseArrays.CHOLMOD.Factor{RC}}(A, b, lambda)
end

function LeastSquares(A::M, b::V, lambda::R=one(R)) where {R <: Real, RC <: Union{R, Complex{R}}, M <: DenseArray{RC, 2}, V <: AbstractArray{RC, 1}}
  LeastSquares{R, RC, M, V, LinAlg.Cholesky{RC, Array{RC, 2}}}(A, b, lambda)
end

function LeastSquares(A::M, b::V, lambda::R=1.0) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractArray{RC, 2}, V <: AbstractArray{RC, 1}}
  warn("Could not infer type of Factorization for $M in LeastSquares, this type will be type-unstable")
  LeastSquares{R, RC, M, V, Factorization}(A, b, lambda)
end

is_convex(f::LeastSquares) = true
is_smooth(f::LeastSquares) = true
is_quadratic(f::LeastSquares) = true

function (f::LeastSquares{R, RC, M, V, F})(x::AbstractArray{D, 1}) where {R, RC, M, V, F, D <: Union{R, Complex{R}}}
  return (f.lambda/2)*vecnorm(f.A*x - f.b, 2)^2
end

function factor_step!(f::LeastSquares{R, RC, M, V, F}, gamma::R) where {R, RC, M, V, F}
  # factor step, two cases: (1) tall A, (2) fat A
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S + I/lamgam)
  f.gamma = gamma
end

function factor_step!(f::LeastSquares{R, RC, M, V, F}, gamma::R) where {R, RC, I, M<:SparseMatrixCSC{RC, I}, V, F}
  # factor step, two cases: (1) tall A, (2) fat A
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S; shift=1.0/lamgam)
  f.gamma = gamma
end

function prox!(y::AbstractArray{D, 1}, f::LeastSquares{R, RC, M, V, F}, x::AbstractArray{D, 1}, gamma::R=one(R)) where {R, RC, M, V, F, D <: Union{R, Complex{R}}}
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
  return (f.lambda/2)*norm(f.A*y-f.b, 2)^2
end

function gradient!(y::AbstractArray{D, 1}, f::LeastSquares{R, RC, M, V, F}, x::AbstractArray{D, 1}) where {R, RC, M, V, F, D <: Union{R, Complex{R}}}
  res = f.A*x - f.b
  Ac_mul_B!(y, f.A, res)
  y .*= f.lambda
  fy = (f.lambda/2)*dot(res, res)
end

fun_name(f::LeastSquares) = "least-squares penalty"
fun_expr(f::LeastSquares) = "x ↦ (λ/2)||A*x - b||^2"
fun_params(f::LeastSquares) = string("λ = $(f.lambda), A = ", typeof(f.A), " of size ", size(f.A), ", b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive(f::LeastSquares, x::AbstractArray, gamma=1.0)
  lamgam = f.lambda*gamma
  y = (f.A'*f.A + I/lamgam)\(f.Atb + x/lamgam)
  fy = (f.lambda/2)*norm(f.A*y-f.b)^2
  return y, fy
end
