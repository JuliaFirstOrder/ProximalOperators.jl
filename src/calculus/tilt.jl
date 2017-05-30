# Tilting (addition with affine function)

"""
  Tilt(f::ProximableFunction, a::AbstractArray, b::Real)

Given function `f`, returns `g(x) = f(x) + <a,x> + b`.
"""

immutable Tilt{T <: ProximableFunction, S <: AbstractArray, R <: Real} <: ProximableFunction
  f::T
  a::S
  b::R
end

is_separable(f::Tilt) = is_separable(f.f)
is_prox_accurate(f::Tilt) = is_prox_accurate(f.f)
is_convex(f::Tilt) = is_convex(f.f)
is_singleton(f::Tilt) = is_singleton(f.f)
is_smooth(f::Tilt) = is_smooth(f.f)
is_quadratic(f::Tilt) = is_quadratic(f.f)
is_generalized_quadratic(f::Tilt) = is_generalized_quadratic(f.f)
is_strongly_convex(f::Tilt) = is_strongly_convex(f.f)

Tilt{T <: ProximableFunction, S <: AbstractArray}(f::T, a::S) = Tilt{T, S, eltype(a)}(f, a, 0.0)

function (g::Tilt){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f(x) + vecdot(g.a, x) + g.b
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::Tilt, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  v = prox!(y, g.f, x - gamma.*g.a, gamma)
  return v + vecdot(g.a, y) + g.b
end

fun_name(f::Tilt) = string("Tilted ", fun_name(f.f))
fun_dom(f::Tilt) = fun_dom(f.f)
fun_expr(f::Tilt) = string(fun_expr(f.f)," + a'x + b")
fun_params(f::Tilt) = "a = $(typeof(f.a)), b = $(f.b)"

function prox_naive{T <: RealOrComplex}(g::Tilt, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  y, v = prox_naive(g.f, x - gamma.*g.a, gamma)
  return y, v + vecdot(g.a, y) + g.b
end
