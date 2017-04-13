# indicator of an affine set

"""
  IndAffine(A::Array{Real,2}, b::Array{Real,1})

Returns the function `g = ind{x : Ax = b}`.

  IndAffine(A::Array{Real,1}, b::Real)

Returns the function `g = ind{x : dot(a,x) = b}`.
"""

immutable IndAffine{T <: RealOrComplex, V} <: IndicatorConvex
  A::AbstractArray{T,2}
  b::AbstractArray{T,1}
  R::V
  function IndAffine(A::AbstractArray{T,2}, b::AbstractArray{T,1})
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrows = vec(sqrt.(sum(abs2.(A), 2)))
    A = (1./normrows).*A # normalize rows of A
    b = (1./normrows).*b # and b accordingly
    if !issparse(A)
      Q, R = qr(A')
      new(A, b, R)
    else
      F = qrfact(A')
      new(A, b, F)
    end
  end
end

IndAffine{T <: RealOrComplex, V<:DenseArray}(A::V, b::AbstractArray{T,1}) =
  IndAffine{T,V}(A, b)

IndAffine{T <: RealOrComplex, V<:SparseMatrixCSC}(A::V, b::AbstractArray{T,1}) =
  IndAffine{T,SparseArrays.SPQR.Factorization{Float64}}(A, b)

IndAffine{T <: RealOrComplex}(a::AbstractArray{T,1}, b::T) =
  IndAffine(a', [b])

function (f::IndAffine){T <: RealOrComplex}(x::AbstractArray{T,1})
  # the tolerance in the following line should be customizable
  if norm(f.A*x - f.b, Inf) <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: RealOrComplex, T2, V<:DenseArray}(f::IndAffine{T2,V}, x::AbstractArray{T,1}, y::AbstractArray{T,1}, gamma::Real=1.0)
  res = f.A*x - f.b
  y[:] = x - f.A'*(f.R\(f.R'\res))
  return 0.0
end

function prox!{T <: RealOrComplex, T2, V<:SparseArrays.SPQR.Factorization}(f::IndAffine{T2,V}, x::AbstractArray{T,1}, y::AbstractArray{T,1}, gamma::Real=1.0)
  RTX_EQUALS_ETB = Int32(3)
  RETX_EQUALS_B  = Int32(1)
  spsolve = SparseArrays.SPQR.solve
  RES = SparseArrays.CHOLMOD.Dense(f.A*x - f.b)
  #We actually store a factor in f.R, not only R
  RRres = convert(typeof(y), spsolve(RETX_EQUALS_B, f.R, spsolve(RTX_EQUALS_ETB, f.R, RES)))
  y .= x .- f.A'RRres
  return 0.0
end

fun_name(f::IndAffine) = "indicator of an affine subspace"
fun_dom(f::IndAffine) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
fun_expr(f::IndAffine) = "x ↦ 0 if Ax = b, +∞ otherwise"
fun_params(f::IndAffine) =
  string( "A = ", typeof(f.A), " of size ", size(f.A), ", ",
          "b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive{T <: RealOrComplex}(f::IndAffine, x::AbstractArray{T,1}, gamma::Real=1.0)
  res = f.A*x - f.b
  y = x - f.A'*(f.R\(f.R'\res))
  return y, 0.0
end
