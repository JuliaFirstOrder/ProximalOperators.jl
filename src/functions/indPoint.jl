# indicator of a point

export IndPoint

"""
**Indicator of a singleton**

    IndPoint(p=0.0)

Returns the indicator of the set
```math
C = \\{p \\}.
```
Parameter `p` can be a scalar, in which case the unique element of `S` has uniform coefficients.
"""

struct IndPoint{T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}} <: ProximableFunction
  p::T
  function IndPoint{T}(p::T) where {T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}}
    new(p)
  end
end

is_separable(f::IndPoint) = true
is_convex(f::IndPoint) = true
is_singleton(f::IndPoint) = true
is_cone(f::IndPoint) = norm(f.p) == 0
is_affine(f::IndPoint) = true

IndPoint(p::T=0.0) where {T <: Union{Real, Complex, AbstractArray{<:RealOrComplex}}} = IndPoint{T}(p)

function (f::IndPoint{R})(x::AbstractArray{T}) where {R <: RealOrComplex, T <: RealOrComplex}
  for k in eachindex(x)
    if abs(x[k] - f.p) > 1e-14
      return +Inf
    end
  end
  return zero(T)
end

function (f::IndPoint{A})(x::AbstractArray{T}) where {A <: AbstractArray, T <: RealOrComplex}
  for k in eachindex(x)
    if abs(x[k] - f.p[k]) > 1e-14
      return +Inf
    end
  end
  return zero(T)
end

function prox!(y::AbstractArray{T}, f::IndPoint{R}, x::AbstractArray{T}, gamma::Real=1.0) where {R <: RealOrComplex, T <: RealOrComplex}
  for k in eachindex(x)
    y[k] = f.p
  end
  return zero(T)
end

function prox!(y::AbstractArray{T}, f::IndPoint{A}, x::AbstractArray{T}, gamma::Real=1.0) where {A <: AbstractArray, T <: RealOrComplex}
  for k in eachindex(x)
    y[k] = f.p[k]
  end
  return zero(T)
end

prox!(y::AbstractArray{T}, f::IndPoint, x::AbstractArray{T}, gamma::AbstractArray) where {T <: RealOrComplex} = prox!(y, f, x, 1.0)

fun_name(f::IndPoint) = "indicator of a point"
fun_dom(f::IndPoint) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndPoint) = "x ↦ 0 if x = p, +∞ otherwise"
fun_params(f::IndPoint) =
  string( "p = ", typeof(f.p) <: AbstractArray ? string(typeof(f.p), " of size ", size(f.p)) : f.p, ", ")

function prox_naive(f::IndPoint, x::AbstractArray{T}, gamma=1.0) where T <: RealOrComplex
    y = similar(x)
    y .= f.p
    return y, 0.0
end
