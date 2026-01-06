export IndGraph

abstract type IndGraph end

"""
    IndGraph(A)

For matrix `A` (dense or sparse) return the indicator function of its graph:
```math
G_A = \\{(x, y) : Ax = y\\}.
```
The evaluation of `prox!` uses direct methods based on LDLt (LL for dense cases) matrix factorization and backsolve.

The `prox!` method operates on pairs `(x, y)` as input/output. So if `f = IndGraph(A)` is the indicator of the graph ``G_A``,
while `(x, y)` and `(c, d)` are pairs of vectors of the same sizes, then
```
prox!((c, d), f, (x, y))
```
writes to `(c, d)` the projection onto ``G_A`` of `(x, y)`.
"""
function IndGraph(A::AbstractMatrix)
    if issparse(A)
        IndGraphSparse(A)
    elseif size(A, 1) > size(A, 2)
        IndGraphSkinny(A)
    else
        IndGraphFat(A)
    end
end

is_convex(f::Type{<:IndGraph}) = true
is_set_indicator(f::Type{<:IndGraph}) = true
is_cone_indicator(f::Type{<:IndGraph}) = true

IndGraph(a::AbstractVector) = IndGraph(a')

# Auxiliary function to be used in fused input call
function splitinput(f::IndGraph, xy)
    @assert length(xy) == f.m + f.n
    x = view(xy, 1:f.n)
    y = view(xy, (f.n + 1):(f.n + f.m))
    return x, y
end

# call additional signatures
function (f::IndGraph)(xy::AbstractVector)
    x, y = splitinput(f, xy)
    return f(x, y)
end
  
(f::IndGraph)(xy::Tuple) = f(xy[1], xy[2])

# prox! additional signatures
function prox!(xy::AbstractVector, f::IndGraph, cd::AbstractVector, gamma)
    x, y = splitinput(f, xy)
    c, d = splitinput(f, cd)
    prox!(x, y, f, c, d)
    return real(eltype(cd))(0)
end

prox!(xy::Tuple, f::IndGraph, cd::Tuple, gamma) = prox!(xy[1], xy[2], f, cd[1], cd[2])

# prox_naive additional signatures
function prox_naive(f::IndGraph, cd::AbstractVector, gamma)
    c, d = splitinput(f, cd)
    x, y, f = prox_naive(f, c, d, gamma)
    return [x;y], f
end

function prox_naive(f::IndGraph, cd::Tuple, gamma)
    x, y, fv = prox_naive(f, cd[1], cd[2], gamma)
    return (x, y), fv
end

include("indGraphSparse.jl")
include("indGraphFat.jl")
include("indGraphSkinny.jl")
