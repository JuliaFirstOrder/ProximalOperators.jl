# indicator of a point

immutable IndPoint{T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}} <: ProximableFunction
  p::T
  function IndPoint{T}(p::T) where {T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}}
    new(p)
  end
end

is_separable(f::IndPoint) = true
is_convex(f::IndPoint) = true
is_cone(f::IndPoint) = norm(f.p) == 0

"""
  IndPoint(p=0.0)

Returns the function `g = ind{x = p}`. Parameter `p` can be
either a scalar or an array of the same dimension as the function argument.
"""

IndPoint{T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}}(p::T=0.0) = IndPoint{T}(p)

function (f::IndPoint{R}){R <: RealOrComplex, T <: RealOrComplex}(x::AbstractArray{T})
  for k in eachindex(x)
    if abs(x[k] - f.p) > 1e-14
      return +Inf
    end
  end
  return 0.0
end

function (f::IndPoint{A}){A <: AbstractArray, T <: RealOrComplex}(x::AbstractArray{T})
  for k in eachindex(x)
    if abs(x[k] - f.p[k]) > 1e-14
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: RealOrComplex, T <: RealOrComplex}(y::AbstractArray{T}, f::IndPoint{R}, x::AbstractArray{T}, gamma::Real=1.0)
  for k in eachindex(x)
    y[k] = f.p
  end
  return 0.0
end

function prox!{A <: AbstractArray, T <: RealOrComplex}(y::AbstractArray{T}, f::IndPoint{A}, x::AbstractArray{T}, gamma::Real=1.0)
  for k in eachindex(x)
    y[k] = f.p[k]
  end
  return 0.0
end

prox!{T <: RealOrComplex}(y::AbstractArray{T}, f::IndPoint, x::AbstractArray{T}, gamma::AbstractArray) = prox!(y, f, x, 1.0)

fun_name(f::IndPoint) = "indicator of a point"
fun_dom(f::IndPoint) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndPoint) = "x ↦ 0 if x = p, +∞ otherwise"
fun_params(f::IndPoint) =
  string( "p = ", typeof(f.p) <: AbstractArray ? string(typeof(f.p), " of size ", size(f.p)) : f.p, ", ")

function prox_naive{T <: RealOrComplex}(f::IndPoint, x::AbstractArray{T}, gamma::Real=1.0)
  return f.p, 0.0
end

prox_naive(f::IndPoint, x::AbstractArray, gamma::AbstractArray) = prox_naive(f, x, 1.0)
