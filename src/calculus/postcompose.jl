# Postcompose with scaling and sum

"""
  Postcompose(f::ProximableFunction, a::Real=1.0, b::Real=0.0)

Returns the function `g(x) = a*f(x) + b`.
"""

immutable Postcompose{T <: ProximableFunction, R <: Real} <: ProximableFunction
  f::T
  a::R
  b::R
  function Postcompose{T,R}(f::T, a::R, b::R) {T <: ProximableFunction, R <: Real}
    if a <= 0.0
      error("parameter a must be positive")
    else
      new(f, a, b)
    end
  end
end

is_separable(f::Postcompose) = is_separable(f.f)
is_prox_accurate(f::Postcompose) = is_prox_accurate(f.f)

Postcompose{T <: ProximableFunction, R <: Real}(f::T, a::R=one(R), b::R=zero(R)) = Postcompose{T, R}(f, a, b)

function (g::Postcompose){T <: RealOrComplex}(x::AbstractArray{T})
  return g.a*g.f(x) + g.b
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::Postcompose, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  v = prox!(y, g.f, x, g.a*gamma)
  return g.a*v + g.b
end

function prox_naive{T <: RealOrComplex}(g::Postcompose, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  y, v = prox_naive(g.f, x, g.a*gamma)
  return y, g.a*v + g.b
end

fun_name(f::Postcompose) = string("Postcomposition of ", fun_name(f.f))
fun_dom(f::Postcompose) = fun_dom(f.f)
fun_expr(f::Postcompose) = "x ↦ a*f(x)+b"
fun_params(f::Postcompose) = string("f(x) = ", fun_expr(f.f), ", a = $(f.a), b = $(f.b)")
