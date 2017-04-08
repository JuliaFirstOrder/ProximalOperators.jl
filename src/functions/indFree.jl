# indicator of the free cone

"""
  IndFree()

Returns the indicator function of the whole space, or "free cone", i.e.,
a function which is identically zero.
"""

immutable IndFree <: ProximableFunction end

is_separable(f::IndFree) = true
is_convex(f::IndFree) = true
is_cone(f::IndFree) = true

"""
  Zero()

Returns the identically zero function.
"""

typealias Zero IndFree

function (f::IndFree)(x::AbstractArray)
  return 0.0
end

function prox!(y::AbstractArray, f::IndFree, x::AbstractArray, gamma::Real=1.0)
  y[:] = x
  return 0.0
end

prox!(y::AbstractArray, f::IndFree, x::AbstractArray, gamma::AbstractArray) = prox!(y, f, x, 1.0)

fun_name(f::IndFree) = "indicator of the free cone"
fun_dom(f::IndFree) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndFree) = "x ↦ 0"
fun_params(f::IndFree) = "none"

function prox_naive(f::IndFree, x::AbstractArray, gamma::Real=1.0)
  return x, 0.0
end

prox_naive(f::IndFree, x::AbstractArray, gamma::AbstractArray) = prox_naive(f, x, 1.0)
