# indicator of a point

immutable IndPoint{T <: Union{Real, Complex, AbstractArray}} <: IndicatorConvex
  p::T
  function IndPoint(p::T)
    new(p)
  end
end

"""
  IndPoint(p=0.0)

Returns the function `g = ind{x = p}`. Parameter `p` can be
either a scalar or an array of the same dimension as the function argument.
"""

IndPoint{T <: Union{Real, Complex, AbstractArray}}(p::T=0.0) = IndPoint{T}(p)

@compat function (f::IndPoint){T <: RealOrComplex}(x::AbstractArray{T})
  if vecnorm(x-f.p, Inf) > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(f::IndPoint, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  y[:] = f.p
  return 0.0
end

fun_name(f::IndPoint) = "indicator of a point"
fun_dom(f::IndPoint) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndPoint) = "x ↦ 0 if x = p, +∞ otherwise"
fun_params(f::IndPoint) =
  string( "p = ", typeof(f.p) <: AbstractArray ? string(typeof(f.p), " of size ", size(f.p)) : f.p, ", ")

function prox_naive{T <: RealOrComplex}(f::IndPoint, x::AbstractArray{T}, gamma::Real=1.0)
  return f.p, 0.0
end

"""
  IndZero()

Returns the function `g = ind{x = 0}`.
"""

IndZero() = IndPoint()
