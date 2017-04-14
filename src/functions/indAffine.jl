# indicator of an affine set

"""
  IndAffine(A::Array{Real,2}, b::Array{Real,1})

Returns the function `g = ind{x : Ax = b}`.

  IndAffine(A::Array{Real,1}, b::Real)

Returns the function `g = ind{x : dot(a,x) = b}`.
"""

immutable IndAffine{T <: RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}, F} <: ProximableFunction
  A::M
  b::V
  R::F
  function IndAffine{T,M,V,F}(A::M, b::V) where {T<:RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}, F}
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
      qlessQR = qrfactqless(A') #Save QR=AE factorization
      new(A, b, qlessQR)
    end
  end
end

IndAffine{T <: RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}}(A::M, b::V) =
  IndAffine{T,M,V,M}(A, b)

IndAffine{T<:RealOrComplex, M<:SparseMatrixCSC, V<:AbstractArray{T,1}}(A::M, b::V) =
  IndAffine{T,M,V,QlessQR{T}}(A, b)

IndAffine{T<:RealOrComplex,V<:AbstractArray{T,1}}(a::V, b::T) =
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

function prox!{R<:Real, T<:RealOrComplex{R}, M<:AbstractSparseArray, V<:AbstractArray{T,1}, F<:QlessQR}(y::V, f::IndAffine{T,M,V,F}, x::V, gamma::R=one(R))
  p = f.R.p   #Sparsity preserving permutation Vector
  FR = f.R.R  #Upper Triangular
  res = f.A*x - f.b
  permute!(res,p)
  A_ldiv_B!(FR,Ac_ldiv_B!(FR,res)) #res = FR\(FR'\res)  #TODO Should probably be done with 
  permute!(res,p)
  y .= x - f.A'*res
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
