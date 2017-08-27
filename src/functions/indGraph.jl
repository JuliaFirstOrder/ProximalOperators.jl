export IndGraph

"""
**Indicator of the graph of a linear operator**

    IndGraph(A)

For matrix `A` (dense or sparse) returns the indicator function of the set
```math
S = \\{(x, y) : Ax = y\\}.
```
The evaluation of `prox!` uses direct methods based on LDLt (LL for dense cases) matrix factorization and backsolve.

The `prox!` method operates on pairs `(x, y)` as input/output. So if `f = IndGraph(A)`,
while `(x, y)` and `(c, d)` are pairs of vectors of the same sizes, then `prox!((c, d), f, (x, y))`
writes to `(c, d)` the projection onto `S` of `(x, y)`.
"""

abstract type IndGraph <: ProximableFunction end

function IndGraph(A::AbstractArray{T,2}) where {T <: RealOrComplex}
  if issparse(A)
    IndGraphSparse(A)
  elseif size(A, 1) > size(A, 2)
    IndGraphSkinny(A)
  else
    IndGraphFat(A)
  end
end

is_convex(f::IndGraph) = true
is_set(f::IndGraph) = true
is_cone(f::IndGraph) = true

IndGraph(a::AbstractArray{T,1}) where {T <: RealOrComplex} =
  IndGraph{T}(a')

# fun_name(f::IndGraph) = "Indicator of an operator graph"
fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
fun_params(f::IndGraph) =
  string( "A = ", typeof(f.A), " of size ", size(f.A))

# Auxiliary function to be used in fused input call
function splitinput(f::IndGraph, xy::AbstractVector{T}) where
    {T <: RealOrComplex}
  @assert length(xy) == f.m + f.n
  x = view(xy, 1:f.n)
  y = view(xy, (f.n + 1):(f.n + f.m))
  return x, y
end

# prox! additional signature
function prox!(
    xy::AbstractVector{T},
    f::IndGraph,
    cd::AbstractVector{T},
    gamma=1.0) where
  {T<:RealOrComplex}
 x, y = splitinput(f, xy)
 c, d = splitinput(f, cd)
 prox!(x, y, f, c, d)
 return 0.0
end

prox!(
    xy::Tuple{AbstractVector{T1}, AbstractVector{T2}},
    f::IndGraph,
    cd::Tuple{AbstractVector{T1}, AbstractVector{T2}},
    gamma=1.0) where
  {T1<:RealOrComplex, T2<:RealOrComplex} = prox!(xy[1], xy[2], f, cd[1], cd[2])

# prox_naive additional signatures
function prox_naive(
    f::IndGraph,
    cd::AbstractVector{T},
    gamma=1.0) where
  {T<:RealOrComplex}
 c, d = splitinput(f, cd)
 x, y, f = prox_naive(f, c, d, gamma)
 return [x;y], f
end

function prox_naive(
    f::IndGraph,
    cd::Tuple{AbstractVector{T1}, AbstractVector{T2}},
    gamma=1.0) where
  {T1<:RealOrComplex, T2<:RealOrComplex}

  x, y, fv = prox_naive(f, cd[1], cd[2], gamma)
  return (x, y), fv
end

include("indGraphSparse.jl")
include("indGraphFat.jl")
include("indGraphSkinny.jl")
