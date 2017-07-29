struct IndGraphSparse{T <: RealOrComplex, Ti <: Number} <: IndGraph
  m::Int
  n::Int
  A::SparseMatrixCSC{T, Ti}
  F::Base.SparseArrays.CHOLMOD.Factor{T} #LDL factorization

  tmp::Array{T, 1}
  tmpx::SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int64}}, true}
  res::Array{T, 1}
end

function IndGraphSparse(A::SparseMatrixCSC{T,Ti}) where
  {T <: RealOrComplex, Ti <: Number}

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

function prox!{T <: RealOrComplex}(
    x::AbstractArray{T, 1},
    y::AbstractArray{T, 1},
    f::IndGraphSparse,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})

  #instead of res = [c + f.A' * d; zeros(f.m)]
  At_mul_B!(f.tmpx, f.A, d)
  f.tmpx .+= c
  # A_ldiv_B!(f.res, f.F, f.tmp) #is not working
  f.res .= f.F \ f.tmp #note here f.tmp which is m+n array
  copy!(x, 1, f.res, 1, f.n)
  copy!(y, 1, f.res, f.n + 1, f.m)
  return 0.0
end

function (f::IndGraphSparse){T <: RealOrComplex}(x::AbstractArray{T,1}, y::AbstractArray{T, 1})
  # the tolerance in the following line should be customizable
  if norm(f.A * x - y, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

fun_name(f::IndGraphSparse) = "Indicator of an operator graph defined by sparse matrix"
# fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive{T <: RealOrComplex}(
    f::IndGraphSparse,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})


  res = [c + f.At * d; zeros(f.m)]
  xy = f.F \ res
  return xy[1:f.n], xy[end - f.m:end], 0.0
end
