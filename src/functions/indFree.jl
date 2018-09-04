# indicator of the free cone

export IndFree

"""
**Indicator of the free cone**

    IndFree()

Returns the indicator function of the whole space, or "free cone", *i.e.*,
a function which is identically zero.
"""
struct IndFree <: ProximableFunction end

is_separable(f::IndFree) = true
is_convex(f::IndFree) = true
is_affine(f::IndFree) = true
is_cone(f::IndFree) = true
is_smooth(f::IndFree) = true
is_quadratic(f::IndFree) = true

const Zero = IndFree

function (f::IndFree)(x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
  return zero(R)
end

function prox!(y::AbstractArray{T}, f::IndFree, x::AbstractArray{T}, gamma::R=one(R)) where {R, T <: RealOrComplex{R}}
  y .= x
  return zero(R)
end

prox!(y::AbstractArray{T}, f::IndFree, x::AbstractArray{T}, gamma::AbstractArray{T}) where {R, T <: RealOrComplex{R}} = prox!(y, f, x)

function gradient!(y::AbstractArray{T}, f::IndFree, x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
  y .= zero(T)
  return zero(R)
end

fun_name(f::IndFree) = "indicator of the free cone"
fun_dom(f::IndFree) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndFree) = "x ↦ 0"
fun_params(f::IndFree) = "none"

function prox_naive(f::IndFree, x::AbstractArray, gamma=1.0)
  return x, 0.0
end
