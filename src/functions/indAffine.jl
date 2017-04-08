# indicator of an affine set

"""
  IndAffine(A::Array{Real,2}, b::Array{Real,1})

Returns the function `g = ind{x : Ax = b}`.

  IndAffine(A::Array{Real,1}, b::Real)

Returns the function `g = ind{x : dot(a,x) = b}`.
"""

<<<<<<< HEAD
immutable IndAffine{T <: RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}} <: ProximableFunction
  A::M
  b::V
  R::M
  function IndAffine{T,M,V}(A::M, b::V) where {T <: RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}}
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrows = vec(sqrt.(sum(abs2.(A), 2)))
    A = (1./normrows).*A # normalize rows of A
    b = (1./normrows).*b # and b accordingly
    Q, R = qr(A')
    new(A, b, R)
  end
end

IndAffine{T <: RealOrComplex, M<:AbstractArray{T,2}, V<:AbstractArray{T,1}}(A::M, b::V) =
  IndAffine{T,M,V}(A, b)

IndAffine{T,V<:AbstractArray{T,1}}(a::V, b::T) =
  IndAffine(reshape(a,1,:), [b])

function (f::IndAffine){T <: RealOrComplex}(x::AbstractArray{T,1})
  # the tolerance in the following line should be customizable
  if norm(f.A*x - f.b, Inf) <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T,1}, f::IndAffine, x::AbstractArray{T,1}, gamma::Real=1.0)
  res = f.A*x - f.b
  y[:] = x - f.A'*(f.R\(f.R'\res))
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
