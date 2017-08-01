# indicator of an affine set

"""
  IndGraph(A::Array{Real,2})

Returns the function `g = ind{(x, y) : Ax = y}`.

  IndGraph(A::Array{Real,1})

Returns the function `g = ind{(x, y) : dot(a,x) = y}`.
"""
# if !isdefined(:RealOrComplex)
  # RealOrComplex = Union{Real, Complex}
# end
#
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

# function (f::IndGraph){T <: RealOrComplex}(x::AbstractArray{T,1}, y::AbstractArray{T, 1})
#   # the tolerance in the following line should be customizable
#   if norm(f.A * x - y, Inf) <= 1e-14
#     return 0.0
#   end
#   return +Inf
# end

# fun_name(f::IndGraph) = "Indicator of an operator graph"
fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
fun_params(f::IndGraph) =
  string( "A = ", typeof(f.A), " of size ", size(f.A))

include("indGraphSparse.jl")
include("indGraphFat.jl")
include("indGraphSkinny.jl")
