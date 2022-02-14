# This implements the variant when A is dense and m < n

using LinearAlgebra

struct IndGraphFat{T} <: IndGraph
  m::Int
  n::Int
  A::Matrix{T}
  AA::Matrix{T}
  tmp::Vector{T}
  # tmpx::Vector{T}
  F::Cholesky{T, Array{T, 2}} # LL factorization
end

function IndGraphFat(A::Matrix{T}) where T
  m, n = size(A)
  AA = A * A'
  F = cholesky(AA + I)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))
  IndGraphFat(m, n, A, AA, Array{T, 1}(undef, m), F)
end

function (f::IndGraphFat)(x, y)
  R = real(eltype(x))
  # the tolerance in the following line should be customizable
  mul!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return R(0)
  end
  return R(Inf)
end

function prox!(x, y, f::IndGraphFat, c, d, gamma=1)
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
  return real(eltype(c))(0)
end

function prox_naive(f::IndGraphFat, c, d, gamma)
  y = f.F \ (f.A * c + f.AA * d)
  return c + f.A' * (d - y), y, real(eltype(c))(0)
end
