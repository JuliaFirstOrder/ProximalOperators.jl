using LinearAlgebra
using SparseArrays
using SuiteSparse

struct IndGraphSparse{T <: RealOrComplex, Ti} <: IndGraph
  m::Int
  n::Int
  A::SparseMatrixCSC{T, Ti}
  F::SuiteSparse.CHOLMOD.Factor{T} #LDL factorization

  tmp::Array{T, 1}
  tmpx::SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int}}, true}
  res::Array{T, 1}
end

function IndGraphSparse(A::SparseMatrixCSC{T,Ti}) where {T <: RealOrComplex, Ti}

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

function prox!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    f::IndGraphSparse,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
  ) where {T <: RealOrComplex}
  #instead of res = [c + f.A' * d; zeros(f.m)]
  mul!(f.tmpx, adjoint(f.A), d)
  f.tmpx .+= c
  # A_ldiv_B!(f.res, f.F, f.tmp) #is not working
  f.res .= f.F \ f.tmp
  # f.res .= f.F \ f.tmp #note here f.tmp which is m+n array
  copyto!(x, 1, f.res, 1, f.n)
  copyto!(y, 1, f.res, f.n + 1, f.m)
  return 0.0
end

function (f::IndGraphSparse)(x::AbstractVector{T}, y::AbstractVector{T}) where
    {T <: RealOrComplex}
  # the tolerance in the following line should be customizable
  tmpy = view(f.res, 1:f.m) # the res is rewritten in prox!
  mul!(tmpy, f.A, x)
  tmpy .-= y
  if norm(tmpy, Inf) <= 1e-12
    return 0.0
  end
  return +Inf
end

function prox_naive(
    f::IndGraphSparse,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1
  ) where {T <: RealOrComplex}

  tmp = f.A'*d
  tmp .+= c
  res = [tmp; zeros(f.m)]
  xy = f.F \ res
  return xy[1:f.n], xy[f.n + 1:f.n + f.m], 0.0
end

function (f::IndGraphSparse)(xy::AbstractVector{T}) where
  {T <: RealOrComplex}
  x, y = splitinput(f, xy)
  return f(x, y)
end

(f::IndGraphSparse)(xy::Tuple{AbstractVector{T}, AbstractVector{T}}) where
  {T <: RealOrComplex} = f(xy[1], xy[2])
