# indicator of the free cone

"""
  IndFree()

Returns the indicator function of the whole space, or "free cone", i.e.,
a function which is identically zero.
"""

immutable IndFree <: IndicatorConvex end

"""
  Zero()

Returns the identically zero function.
"""

typealias Zero IndFree

@compat function (f::IndFree)(x::AbstractArray)
  return 0.0
end

function prox!(f::IndFree, x::AbstractArray, y::AbstractArray, gamma::Real=1.0)
  y[:] = x
  return 0.0
end

fun_name(f::IndFree) = "indicator of the free cone"
fun_dom(f::IndFree) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndFree) = "x ↦ 0"
fun_params(f::IndFree) = "none"

function prox_naive(f::IndFree, x::AbstractArray, gamma::Real=1.0)
  return x, 0.0
end
