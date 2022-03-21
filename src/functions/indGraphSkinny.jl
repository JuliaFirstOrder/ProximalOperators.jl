# This implements the variant when A is dense and m > n

using LinearAlgebra

struct IndGraphSkinny{T} <: IndGraph
  m::Int
  n::Int
  A::Matrix{T}
  AA::Matrix{T}
  F::Cholesky{T, Matrix{T}} #LL factorization
  tmp::Vector{T}
end

function IndGraphSkinny(A::Matrix{T}) where T
  m, n = size(A)
  AA = A' * A
  F = cholesky(AA + I)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))
  #The tmp vector assumes that difference between m and n is not drastic.
  # If someone will decide to solve it using m >> n, then tmp vector might be
  # considered to have only n position required to prox esitmation and indicator
  # calculation might be converted to less efficient.
  tmp = Vector{T}(undef, m)
  IndGraphSkinny(m, n, A, AA, F, tmp)
end

function (f::IndGraphSkinny)(x, y)
  R = real(eltype(x))
  # the tolerance in the following line should be customizable
  mul!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return R(0)
  end
  return R(Inf)
end

function prox!(x, y, f::IndGraphSkinny, c, d, gamma=1)
  # x[:] = f.F \ (c + f.A' * d)
  mul!(x, adjoint(f.A), d)
  x .+= c
  ldiv!(f.F, x)
  mul!(y, f.A, x)
  return real(eltype(c))(0)
end

function prox_naive(f::IndGraphSkinny, c, d, gamma)
  x = f.F \ (c + f.A' * d)
  return x, f.A * x, real(eltype(c))(0)
end
