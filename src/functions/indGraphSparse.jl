using LinearAlgebra
using SparseArrays
using SuiteSparse

struct IndGraphSparse{T, Ti} <: IndGraph
  m::Int
  n::Int
  A::SparseMatrixCSC{T, Ti}
  F::SuiteSparse.CHOLMOD.Factor{T} #LDL factorization
  tmp::Vector{T}
  tmpx::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
  res::Vector{T}
end

function IndGraphSparse(A::SparseMatrixCSC{T,Ti}) where {T, Ti}
  m, n = size(A)
  K = [SparseMatrixCSC{T}(I, n, n) A'; A -SparseMatrixCSC{T}(I, m, m)]
  F = ldlt(K)
  tmp = Array{T, 1}(undef, m + n)
  tmpx = view(tmp, 1:n)
  tmpy = view(tmp, (n + 1):(n + m)) #second part is always zeros
  fill!(tmpy, 0)
  res = Array{T,1}(undef, m + n)
  return IndGraphSparse(m, n, A, F, tmp, tmpx, res)
end

function (f::IndGraphSparse)(x, y)
  R = real(eltype(x))
  # the tolerance in the following line should be customizable
  tmpy = view(f.res, 1:f.m) # the res is rewritten in prox!
  mul!(tmpy, f.A, x)
  tmpy .-= y
  if norm(tmpy, Inf) <= 1e-12
    return R(0)
  end
  return R(Inf)
end

function prox!(x, y, f::IndGraphSparse, c, d, gamma=1)
  #instead of res = [c + f.A' * d; zeros(f.m)]
  mul!(f.tmpx, adjoint(f.A), d)
  f.tmpx .+= c
  # A_ldiv_B!(f.res, f.F, f.tmp) #is not working
  f.res .= f.F \ f.tmp
  # f.res .= f.F \ f.tmp #note here f.tmp which is m+n array
  copyto!(x, 1, f.res, 1, f.n)
  copyto!(y, 1, f.res, f.n + 1, f.m)
  return real(eltype(c))(0)
end

function prox_naive(f::IndGraphSparse, c, d, gamma)
  tmp = f.A'*d
  tmp .+= c
  res = [tmp; zeros(f.m)]
  xy = f.F \ res
  return xy[1:f.n], xy[f.n + 1:f.n + f.m], real(eltype(c))(0)
end
