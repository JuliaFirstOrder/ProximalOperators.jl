# indicator of the zero cone

"""
  IndZero()

Returns the indicator function of the zero point, or "zero cone", i.e.,
  `g(x) = 0 if x = 0, +∞ otherwise`
"""

immutable IndZero <: IndicatorConvexCone end

function (f::IndZero){T <: RealOrComplex}(x::AbstractArray{T})
  for k in eachindex(x)
    if x[k] != zero(T)
      return Inf
    end
  end
  return Inf
end

function prox!{T <: RealOrComplex}(f::IndZero, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  for k in eachindex(x)
    y[k] = zero(T)
  end
  return 0.0
end

fun_name(f::IndZero) = "indicator of the zero cone"
fun_dom(f::IndZero) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndZero) = "x ↦ 0 if all(x = 0), +∞ otherwise"
fun_params(f::IndZero) = "none"

function prox_naive(f::IndZero, x::AbstractArray, gamma::Real=1.0)
  return zero(x), 0.0
end
