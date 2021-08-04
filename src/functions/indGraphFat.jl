# This implements the variant when A is dense and m < n

using LinearAlgebra

struct IndGraphFat{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::Array{T,2}
  AA::Array{T,2}
  tmp::Array{T,1}
  # tmpx::Array{T,1}
  F::Cholesky{T, Array{T, 2}} # LL factorization
end

function IndGraphFat(A::Array{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A * A'

  F = cholesky(AA + I)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  IndGraphFat(m, n, A, AA, Array{T, 1}(undef, m), F)
end

function prox!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    f::IndGraphFat,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
    ) where {T <: RealOrComplex}

  # y .= f.F \ (f.A * c + f.AA * d)
  mul!(f.tmp, f.A, c)
  mul!(y, f.AA, d)
  y .+= f.tmp
  ldiv!(f.F, y)

  # f.A' * (d - y) + c # note: for complex the complex conjugate is used
  copyto!(f.tmp, d)
  f.tmp .-= y
  mul!(x, adjoint(f.A), f.tmp)
  x .+= c
  return 0.0
end

function (f::IndGraphFat)(x::AbstractVector{T}, y::AbstractVector{T}) where
    {T <: RealOrComplex}
  # the tolerance in the following line should be customizable
  mul!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

function prox_naive(
    f::IndGraphFat,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
  ) where {T <: RealOrComplex}

  y = f.F \ (f.A * c + f.AA * d)
  return c + f.A' * (d - y), y, 0.0
end

function (f::IndGraphFat)(xy::AbstractVector{T}) where
  {T <: RealOrComplex}
  x, y = splitinput(f, xy)
  return f(x, y)
end

(f::IndGraphFat)(xy::Tuple{AbstractVector{T}, AbstractVector{T}}) where
  {T <: RealOrComplex} = f(xy[1], xy[2])
