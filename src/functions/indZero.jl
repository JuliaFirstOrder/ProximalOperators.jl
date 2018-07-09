# indicator of the zero cone

export IndZero

"""
**Indicator of the zero cone**

    IndZero()

Returns the indicator function of the set containing the origin, the "zero cone".
"""

struct IndZero <: ProximableFunction end

is_separable(f::IndZero) = true
is_convex(f::IndZero) = true
is_singleton(f::IndZero) = true
is_cone(f::IndZero) = true
is_affine(f::IndZero) = true

function (f::IndZero)(x::AbstractArray{T}) where T <: RealOrComplex
  for k in eachindex(x)
    if x[k] != zero(T)
      return Inf
    end
  end
  return 0.0
end

function prox!(y::AbstractArray{T}, f::IndZero, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
  for k in eachindex(x)
    y[k] = zero(T)
  end
  return 0.0
end

prox!(y::AbstractArray{T}, f::IndZero, x::AbstractArray{T}, gamma::AbstractArray) where {T <: RealOrComplex} = prox!(y, f, x, 1.0)

fun_name(f::IndZero) = "indicator of the zero cone"
fun_dom(f::IndZero) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndZero) = "x ↦ 0 if all(x = 0), +∞ otherwise"
fun_params(f::IndZero) = "none"

function prox_naive(f::IndZero, x::AbstractArray, gamma::Real=1.0)
  return zero(x), 0.0
end

prox_naive(f::IndZero, x::AbstractArray, gamma::AbstractArray) = prox_naive(f, x, 1.0)
