# indicator of an affine set

"""
  IndGraph(A::Array{Real,2})

Returns the function `g = ind{(x, y) : Ax = y}`.

  IndGraph(A::Array{Real,1})

Returns the function `g = ind{(x, y) : dot(a,x) = y}`.
"""
# if !isdefined(:RealOrComplex)
  RealOrComplex = Union{Real, Complex}
# end
#
struct IndGraph{T <: RealOrComplex} <: ProximableFunction
  m::Int
  n::Int
  A::AbstractArray{T,2}
  At::AbstractArray{T,2}
  F::Base.SparseArrays.CHOLMOD.Factor{Float64}
  # A = @view K[end - m: end, 1:n]
  # At = @view K[1:n, end-m:end]
  function IndGraph(A::AbstractArray{T,2})
    if !issparse(A)
      error("Not implemented")
    end
    m, n = size(A)
    At = A'
    K = [speye(n) At; A -speye(m)]

    F = LinAlg.ldltfact(K)
    #normrows = vec(sqrt.(sum(abs2.(A), 2)))

    new(m, n, A, At, F)
  end
end

is_convex(f::IndGraph) = true
is_set(f::IndGraph) = true
is_cone(f::IndGraph) = false

IndGraph{T <: RealOrComplex}(A::AbstractArray{T,2}) =
  IndGraph{T}(A)

IndGraph{T <: RealOrComplex}(a::AbstractArray{T,1}) =
  IndGraph{T}(a')

function (f::IndGraph){T <: RealOrComplex}(x::AbstractArray{T,1}, y::AbstractArray{T, 1})
  # the tolerance in the following line should be customizable
  if norm(f.A * x - y, Inf) <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: RealOrComplex}(
    x::AbstractArray{T, 1},
    y::AbstractArray{T, 1},
    f::IndGraph,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})
  res = [c + f.At * d; zeros(f.m)]
  xy = f.F \ res
  x[:] = xy[1:f.n]
  y[:] = xy[end - f.m:end]
  return 0.0
end

fun_name(f::IndGraph) = "indicator of an operator graph"
fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
fun_params(f::IndGraph) =
  string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive{T <: RealOrComplex}(
    f::IndGraph,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})
  res = [c + f.At * d; zeros(f.m)]
  xy = f.F \ res
  return xy[1:f.n], xy[end - f.m:end], 0.0
end
