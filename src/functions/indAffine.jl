# indicator of an affine set

"""
  IndAffine(A::Array{Float64,2}, b::Array{Float64,1})

Returns the function `g = ind{x : Ax = b}`.

  IndAffine(A::Array{Float64,1}, b::Float64)

Returns the function `g = ind{x : dot(a,x) = b}`.
"""

immutable IndAffine <: IndicatorConvex
  A::RealOrComplexMatrix
  b::RealOrComplexVector
  R::RealOrComplexMatrix
  function IndAffine(A::RealOrComplexMatrix, b::RealOrComplexVector)
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrows = vec(sqrt(sum(abs(A).^2, 2)))
    A = (1./normrows).*A # normalize rows of A
    b = (1./normrows).*b # and b accordingly
    Q, R = qr(A')
    new(A, b, R)
  end
  IndAffine(a::RealOrComplexVector, b::RealOrComplex) =
    IndAffine(a', [b])
end

@compat function (f::IndAffine)(x::RealOrComplexVector)
  # the tolerance in the following line should be customizable
  if norm(f.A*x - f.b, Inf) <= 1e-14 return 0.0 end
  return +Inf
end

function prox!(f::IndAffine, x::RealOrComplexVector, gamma::Float64, y::RealOrComplexVector)
  res = f.A*x - f.b
  y[:] = x - f.A'*(f.R\(f.R'\res))
  return 0.0
end

fun_name(f::IndAffine) = "indicator of an affine subspace"
fun_type(f::IndAffine) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(f::IndAffine) = "x ↦ 0 if Ax = b, +∞ otherwise"
fun_params(f::IndAffine) =
  string( "A = ", typeof(f.A), " of size ", size(f.A), ", ",
          "b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive(f::IndAffine, x::RealOrComplexVector, gamma::Float64=1.0)
  res = f.A*x - f.b
  y = x - f.A'*(f.R\(f.R'\res))
  return y, 0.0
end
