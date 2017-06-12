# least squares penalty

export LeastSquares

"""
  LeastSquares(A, b, λ=1.0)

Returns the function `f(x) = (λ/2)⋅||Ax-b||^2`.
"""
type LeastSquares{RC <: RealOrComplex, R <: Real, M <: AbstractArray{RC, 2}, V <: AbstractArray{RC, 1}, F <: Factorization} <: ProximableFunction
  A::M
  b::V
  Atb::V
  lambda::R
  gamma::R
  S::M
  U::F
  function LeastSquares{RC,R,M,V,F}(A::M, b::V, lambda::R) where {RC <: RealOrComplex, R<:Real, M<:AbstractArray{RC,2}, V<:AbstractArray{RC,1}, F<:Factorization}
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

function LeastSquares(A::M, b::V, lambda::R) where {RC <: RealOrComplex, R<:Real, I<:Integer, M<:SparseMatrixCSC{RC,I}, V<:AbstractArray{RC,1}}
  LeastSquares{RC,R,M,V,SparseArrays.CHOLMOD.Factor{RC}}(A,b,lambda)
end

function LeastSquares(A::M, b::V, lambda::R) where {RC <: RealOrComplex, R<:Real, M<:DenseArray{RC,2}, V<:AbstractArray{RC,1}}
  LeastSquares{RC,R,M,V,LinAlg.Cholesky{RC,Array{RC,2}}}(A,b,lambda)
end

function LeastSquares(A::M, b::V, lambda::R) where {RC <: AbstractArray, R<:Real, M<:AbstractArray{RC,2}, V<:AbstractArray{RC,1}}
  warn("Could not infer type of Factorization for $M in LeastSquares, this type will be type-unstable")
  LeastSquares{RC,R,M,V,Factorization}(A,b,lambda)
end

is_convex(f::LeastSquares) = true
is_smooth(f::LeastSquares) = true
is_quadratic(f::LeastSquares) = true

LeastSquares{RC <: RealOrComplex, R<:Real, M<:AbstractArray{RC,2}, V<:AbstractArray{RC,1}}(A::M, b::V, lambda::R=1.0) =
  LeastSquares{RC,R,M,V}(A, b, lambda)

function (f::LeastSquares{RC,R,M,V,F}){RC, R, M, V, F}(x::AbstractArray{RC,1})
  return (f.lambda/2)*vecnorm(f.A*x - f.b, 2)^2
end

function factor_step!{RC, R, M, V ,F}(f::LeastSquares{RC,R,M,V,F}, gamma::R)
  # factor step, two cases: (1) tall A, (2) fat A
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S + I/lamgam)
  f.gamma = gamma
end

function factor_step!{RC <: RealOrComplex, R, I<:Integer, M<:SparseMatrixCSC{RC,I}, V, F}(f::LeastSquares{RC,R,M,V,F}, gamma::R)
  # factor step, two cases: (1) tall A, (2) fat A
  lamgam = f.lambda*gamma
  f.U = cholfact(f.S; shift=1.0/lamgam)
  f.gamma = gamma
end

#R<:Real needed to avoid ambiguity
function prox!{RC,R<:Real,M,V,F}(y::AbstractArray{RC,1}, f::LeastSquares{RC,R,M,V,F}, x::AbstractArray{RC,1}, gamma::R=one(R))
  # if gamma different from f.gamma then call factor_step!
  if gamma != f.gamma
    factor_step!(f, gamma)
  end
  lamgam = f.lambda*gamma
  # solve step, two cases: (1) tall A, (2) fat A
  q = f.Atb + x/lamgam
  if size(f.A,1) >= size(f.A,2)
    y[:] = f.U\q
  else
    y[:] = lamgam*(q - (f.A'*(f.U\(f.A*q))))
  end
  return (f.lambda/2)*norm(f.A*y-f.b, 2)^2
end

fun_name(f::LeastSquares) = "least-squares penalty"
fun_dom{R <: Real}(f::LeastSquares{R}) = "AbstractArray{Real,1}"
fun_dom{C <: Complex}(f::LeastSquares{C}) = "AbstractArray{Complex,1}"
fun_expr(f::LeastSquares) = "x ↦ (λ/2)||A*x - b||^2"
fun_params(f::LeastSquares) = string("λ = $(f.lambda), A = ", typeof(f.A), " of size ", size(f.A), ", b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive{R <: RealOrComplex}(f::LeastSquares, x::AbstractArray{R,1}, gamma::Real=1.0)
  lamgam = f.lambda*gamma
  y = (f.A'*f.A + I/lamgam)\(f.Atb + x/lamgam)
  fy = (f.lambda/2)*norm(f.A*y-f.b)^2
  return y, fy
end
