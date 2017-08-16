export IndGraphSparse

struct IndGraphSparse{T <: RealOrComplex, Ti} <: IndGraph
  m::Int
  n::Int
  A::SparseMatrixCSC{T, Ti}
  F::Base.SparseArrays.CHOLMOD.Factor{T} #LDL factorization

  tmp::Array{T, 1}
  tmpx::SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int}}, true}
  res::Array{T, 1}
end

function IndGraphSparse(A::SparseMatrixCSC{T,Ti}) where
  {T <: RealOrComplex, Ti}

  m, n = size(A)
  K = [speye(n) A'; A -speye(m)]

  F = LinAlg.ldltfact(K)

  tmp = Array{T,1}(m + n)
  tmpx = view(tmp, 1:n)
  tmpy = view(tmp, (n + 1):(n + m)) #second part is always zeros
  fill!(tmpy, 0)

  res = Array{T,1}(m + n)
  return IndGraphSparse(m, n, A, F, tmp, tmpx, res)
end

# is_convex(f::IndGraph) = true
# is_set(f::IndGraph) = true
# is_cone(f::IndGraph) = false

function prox!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    f::IndGraphSparse,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1.0
  ) where {T <: RealOrComplex}
  #instead of res = [c + f.A' * d; zeros(f.m)]
  At_mul_B!(f.tmpx, f.A, d)
  f.tmpx .+= c
  # A_ldiv_B!(f.res, f.F, f.tmp) #is not working
  f.res .= f.F \ f.tmp
  # f.res .= f.F \ f.tmp #note here f.tmp which is m+n array
  copy!(x, 1, f.res, 1, f.n)
  copy!(y, 1, f.res, f.n + 1, f.m)
  return 0.0
end

function (f::IndGraphSparse)(x::AbstractVector{T}, y::AbstractVector{T}) where
    {T <: RealOrComplex}
  # the tolerance in the following line should be customizable
  tmpy = view(f.res, 1:f.m) # the res is rewritten in prox!
  A_mul_B!(tmpy, f.A, x)
  tmpy .-= y
  if norm(tmpy, Inf) <= 1e-12
    return 0.0
  end
  return +Inf
end

fun_name(f::IndGraphSparse) = "Indicator of an operator graph defined by sparse matrix"
# fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive(
    f::IndGraphSparse,
    c::AbstractVector{T},
    d::AbstractVector{T},
    gamma=1.0
  ) where {T <: RealOrComplex}

  tmp = At_mul_B(f.A, d);
  tmp .+= c;
  res = [tmp; zeros(f.m)]
  xy = f.F \ res
  return xy[1:f.n], xy[f.n + 1:f.n + f.m], 0.0
end

## Additional signatures
# prox!(xy::Tuple{AbstractVector{T},AbstractVector{T}},
#       f::IndGraphSparse,
#       cd::Tuple{AbstractVector{T},AbstractVector{T}}, gamma=1.0
#   ) where {T <: RealOrComplex} =
#     prox!(xy[1], xy[2], f, cd[1], cd[2])
#
function (f::IndGraphSparse)(xy::AbstractVector{T}) where
  {T <: RealOrComplex}
  x, y = splitinput(f, xy)
  return f(x, y)
end
