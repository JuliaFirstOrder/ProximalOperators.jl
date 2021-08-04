# This implements the variant when A is dense and m > n

using LinearAlgebra

struct IndGraphSkinny{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::Array{T, 2}
  AA::Array{T, 2}
  F::Cholesky{T, Array{T, 2}} #LL factorization
  tmp::Array{T, 1}
end

function IndGraphSkinny(A::Array{T, 2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A' * A

  F = cholesky(AA + I)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  #The tmp vector assumes that difference between m and n is not drastic.
  # If someone will decide to solve it using m >> n, then tmp vector might be
  # considered to have only n position required to prox esitmation and indicator
  # calculation might be converted to less efficient.
  tmp = Array{T, 1}(undef, m)
  IndGraphSkinny(m, n, A, AA, F, tmp)
end

function (f::IndGraphSkinny)(x::AbstractVector{T}, y::AbstractVector{T}) where
    {T <: RealOrComplex}
  # the tolerance in the following line should be customizable
  mul!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

function prox!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    f::IndGraphSkinny,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
    ) where {T <: RealOrComplex}

  # x[:] = f.F \ (c + f.A' * d)
  mul!(x, adjoint(f.A), d)
  x .+= c
  ldiv!(f.F, x)

  mul!(y, f.A, x)
  return 0.0
end

function prox_naive(
    f::IndGraphSkinny,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
    ) where {T <: RealOrComplex}

  x = f.F \ (c + f.A' * d)
  return x, f.A * x, 0.0
end

function (f::IndGraphSkinny)(xy::AbstractVector{T}) where
  {T <: RealOrComplex}
  x, y = splitinput(f, xy)
  return f(x, y)
end

(f::IndGraphSkinny)(xy::Tuple{AbstractVector{T}, AbstractVector{T}}) where
  {T <: RealOrComplex} = f(xy[1], xy[2])
