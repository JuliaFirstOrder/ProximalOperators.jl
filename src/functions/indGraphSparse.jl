struct IndGraphSparse{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::AbstractArray{T,2}
  F::Base.SparseArrays.CHOLMOD.Factor{T} #LDL factorization
end

function IndGraphSparse(A::AbstractArray{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  K = [speye(n) A'; A -speye(m)]

  F = LinAlg.ldltfact(K)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  return IndGraphSparse(m, n, A, F)
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
  res = [c + f.A' * d; zeros(f.m)]
  xy = f.F \ res
  x[:] = xy[1:f.n]
  y[:] = xy[end - f.m + 1:end]
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
