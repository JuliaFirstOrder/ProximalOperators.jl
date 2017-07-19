# indicator of an affine set

export IndAffine

"""
**Indicator of an affine subspace**

    IndAffine(A, b)

If `A` is a matrix (dense or sparse) and `b` is a vector, returns the indicator function of the set
```math
S = \\{x : Ax = b\\}.
```
If `A` is a vector and `b` is a scalar, returns the indicator function of the set
```math
S = \\{x : \\langle A, x \\rangle = b\\}.
```
"""

immutable IndAffine{T <: RealOrComplex, M <: AbstractArray{T, 2}, V <: AbstractArray{T, 1}, F} <: ProximableFunction
  A::M
  b::V
  R::F
  function IndAffine{T,M,V,F}(A::M, b::V) where {T<:RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}, F}
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrowsinv = 1./vec(sqrt.(sum(abs2.(A), 2)))
    A = normrowsinv.*A # normalize rows of A
    b = normrowsinv.*b # and b accordingly
    if !issparse(A)
      Q, R = qr(A')
      new(A, b, R)
    else
      RF = qrfact(A') #Save QR=AE factorization
      new(A, b, RF)
    end
  end
end

is_affine(f::IndAffine) = true
is_cone(f::IndAffine) = norm(f.b) == 0.0
is_generalized_quadratic(f::IndAffine) = true

IndAffine{T <: RealOrComplex, M <: AbstractArray{T, 2}, V <: AbstractArray{T, 1}}(A::M, b::V) =
  IndAffine{T, M, V, M}(A, b)

IndAffine{T <: RealOrComplex, M <: SparseMatrixCSC, V <: AbstractArray{T, 1}}(A::M, b::V) =
  IndAffine{T, M, V, SparseArrays.SPQR.Factorization{T}}(A, b)

IndAffine{T <: RealOrComplex, V <: AbstractArray{T, 1}}(a::V, b::T) =
  IndAffine(reshape(a,1,:), [b])

function (f::IndAffine){R<:Real, T <: RealOrComplex{R}}(x::AbstractArray{T,1})
  # the tolerance in the following line should be customizable
  if norm(f.A*x - f.b, Inf) <= 1e-14
    return zero(R)
  end
  return typemax(R)
end

function prox!{R<:Real, T<:RealOrComplex{R}, M<:DenseArray, V<:AbstractArray{T,1}}(y::V, f::IndAffine{T,M,V,M}, x::V, gamma::R=one(R))
  res = f.A*x - f.b
  y[:] = x - f.A'*(f.R\(f.R'\res))
  return zero(R)
end

function prox!{R<:Real, T<:RealOrComplex{R}, M<:AbstractSparseArray, V<:AbstractArray{T,1}, F<:SparseArrays.SPQR.Factorization}(y::V, f::IndAffine{T,M,V,F}, x::V, gamma::R=one(R))
  RTX_EQUALS_ETB = Int32(3)
  RETX_EQUALS_B  = Int32(1)
  spsolve = SparseArrays.SPQR.solve
  RES = SparseArrays.CHOLMOD.Dense(f.A*x - f.b)
  # QR=AE so tmp = R'\E'res, RRres = E*R\(tmp), i.e. RRres = E*R\(R'\(E'res))
  RRres = convert(typeof(y), spsolve(RETX_EQUALS_B, f.R, spsolve(RTX_EQUALS_ETB, f.R, RES)))
  y .= x .- f.A'RRres
  return zero(R)
end

fun_name(f::IndAffine) = "indicator of an affine subspace"
fun_dom(f::IndAffine) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
fun_expr(f::IndAffine) = "x ↦ 0 if Ax = b, +∞ otherwise"
fun_params(f::IndAffine) =
  string( "A = ", typeof(f.A), " of size ", size(f.A), ", ",
          "b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive{R<:Real, T <: RealOrComplex{R}}(f::IndAffine, x::AbstractArray{T,1}, gamma::R=one(R))
  res = f.A*x - f.b
  y = x - f.A'*(f.R\(f.R'\res))
  return y, zero(R)
end

#Not really naive, comparison is done to non-sparse in tests
function prox_naive{R<:Real, T <: RealOrComplex{R}, M<:AbstractSparseArray}(f::IndAffine{T,M}, x::AbstractArray{T,1}, gamma::R=one(R))
  res = SparseArrays.CHOLMOD.Dense(f.A*x - f.b)
  y = x - f.A'*convert(typeof(x), SparseArrays.SPQR.solve(Int32(1), f.R, SparseArrays.SPQR.solve(Int32(3), f.R, res)))
  return y, zero(R)
end
