# Tilting (addition with affine function)

immutable Tilt{T <: ProximableFunction, S <: AbstractArray, R <: Real} <: ProximableFunction
  f::T
  a::S
  b::R
end

Tilt{T <: ProximableFunction, S <: AbstractArray}(f::T, a::S) = Tilt{T, S, eltype(a)}(f, a, 0.0)

function (g::Tilt){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f(x) + vecdot(g.a, x) + g.b
end

function prox!{T <: RealOrComplex}(g::Tilt, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  v = prox!(g.f, x - gamma*g.a, y, gamma)
  return v + vecdot(g.a, y) + g.b
end

function prox_naive{T <: RealOrComplex}(g::Tilt, x::AbstractArray{T}, gamma::Real=1.0)
  y, v = prox_naive(g.f, x - gamma*g.a, gamma)
  return y, v + vecdot(g.a, y) + g.b
end

is_prox_accurate(f::Tilt) = is_prox_accurate(f.f)

fun_name(f::Tilt) = string("Tilted ", fun_name(f.f))
fun_dom(f::Tilt) = fun_dom(f.f)
fun_expr(f::Tilt) = string(fun_expr(f.f)," + a'x")
fun_params(f::Tilt) = string(fun_expr(f.f), ", a = $(typeof(f.a))")
