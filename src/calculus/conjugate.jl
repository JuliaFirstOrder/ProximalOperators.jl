# Conjugate

"""
  Conjugate(f::ProximableFunction)

For a function f returns the conjugate function f*, defined as

  f*(y) = sup_x {<x,y> - f(x)}.
"""

immutable Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
end

function prox!{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  # Moreau identity
  v = prox!(g.f, x/gamma, y, 1.0/gamma)
  v = vecdot(x,y) - gamma*vecdot(y,y) - v
  y[:] *= -gamma
  y[:] += x
  return v
end

fun_name(g::Conjugate) = string("conjugate function of ", typeof(g.f))
fun_type(g::Conjugate) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(g::Conjugate) = "x ↦ sup_y { <x,y> - f(y) } "
fun_params(g::Conjugate) = string("f = ", typeof(g.f))
