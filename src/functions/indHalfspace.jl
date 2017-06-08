# indicator of a halfspace

export IndHalfspace

"""
  IndHalfspace(a::Array{Real}, b::Real)

Returns the function `g = ind{x : <a,x> ⩽ b}`.
"""

immutable IndHalfspace{R <: Real, T <: AbstractVector{R}} <: ProximableFunction
  a::T
  b::R
  function IndHalfspace{R,T}(a::T, b::R) where {R <: Real, T <: AbstractVector{R}}
    norma = vecnorm(a)
    new(a/norma, b/norma)
  end
end

is_convex(f::IndHalfspace) = true
is_set(f::IndHalfspace) = true
is_cone(f::IndHalfspace) = (f.b == 0)

IndHalfspace{R <: Real, T <: AbstractVector{R}}(a::T, b::R) = IndHalfspace{R, T}(a, b)

function (f::IndHalfspace){T <: Real}(x::AbstractArray{T})
  s = vecdot(f.a,x)-f.b
  if s <= 1e-14
    return 0.0
  end
  return +Inf
end

function prox!{T <: Real}(y::AbstractArray{T}, f::IndHalfspace, x::AbstractArray{T}, gamma::Real=1.0)
  s = vecdot(f.a,x)-f.b
  if s <= 0
    y[:] = x
    return 0.0
  end
  for k in eachindex(x)
    y[k] = x[k] - s*f.a[k]
  end
  return 0.0
end

fun_name(f::IndHalfspace) = "indicator of a halfspace"
fun_dom(f::IndHalfspace) = "AbstractArray{Real}"
fun_expr(f::IndHalfspace) = "x ↦ 0 if <a,x> ⩽ b, +∞ otherwise"
fun_params(f::IndHalfspace) =
  string( "a = ", typeof(f.a), " of size ", size(f.a), ", ",
          "b = $(f.b)")

function prox_naive{T <: Real}(f::IndHalfspace, x::AbstractArray{T}, gamma::Real=1.0)
  s = vecdot(f.a,x)-f.b
  if s <= 0
    return x, 0.0
  end
  return x - s*f.a, 0.0
end
