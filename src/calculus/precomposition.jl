# Precomposition with uniform scaling and translation

"""
  Precomposition(f::ProximableFunction, a::Real=1.0, b::Real=0.0)

For a function f, returns g(x) = f(a*x + b).
"""

immutable Precomposition{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Real} <: ProximableFunction
  f::T
  a::R
  b::S
  function Precomposition(f::T, a::R, b::S=0.0)
    if any(a == 0.0)
      error("elements of a must be nonzero")
    else
      Precomposition{T, R, S}(f, a, b)
    end
  end
end

Precomposition{T <: ProximableFunction, R <: AbstractArray, S <: Real}(f::T, a::R, b::S=0.0) = Precomposition{T, R, S}(f, a, b)

Precomposition{T <: ProximableFunction, S <: Real}(f::T, a::S=1.0, b::S=0.0) = Precomposition{T, S, S}(f, a, b)

@compat function (g::Precomposition){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f(g.a*x + g.b)
end

function prox!{T <: RealOrComplex}(g::Precomposition, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  y[:] = g.a * x + g.b
  v = prox!(g.f, y, y, (g.a * g.a) * gamma)
  y[:] -= g.b
  y[:] /= g.a
  return v
end

fun_name(g::Precomposition) = string(typeof(g.f), "composed with scaling and translation")
fun_type(g::Precomposition) = fun_type(g.f)
fun_expr(g::Precomposition) = "x ↦ f(a*x + b)"
fun_params(g::Precomposition) = string("f = ", typeof(g.f), ", a = $(g.a)", ", b = $(g.b)")
