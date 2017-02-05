# Precomposition with scaling and translation

"""
  Precomposition(f::ProximableFunction, a::AbstractArray, b::AbstractArray)

Returns the function `g(x) = f(diag(a)*x) + b`. Function `f` must be separable, or `a` must be a scalar, for the `prox` of `g` to be computable.
"""

immutable Precomposition{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  a::R
  b::S
  function Precomposition(f::T, a::R, b::S)
    if !(eltype(a) <: Real && eltype(b) <: Real)
      error("a and b must be real")
    end
    if any(a == 0.0)
      error("elements of a must be nonzero")
    else
      new(f, a, b)
    end
  end
end

Precomposition{T <: ProximableFunction, S <: Real}(f::T, a::S=1.0, b::S=0.0) = Precomposition{T, S, S}(f, a, b)

Precomposition{T <: ProximableFunction, R <: AbstractArray, S <: Real}(f::T, a::R, b::S=1.0) = Precomposition{T, R, S}(f, a, b)

Precomposition{T <: ProximableFunction, R <: Union{AbstractArray, Real}, S <: AbstractArray}(f::T, a::R, b::S) = Precomposition{T, R, S}(f, a, b)

is_prox_accurate(f::Precomposition) = is_prox_accurate(f.f)

function (g::Precomposition{T, S, V}){T <: ProximableFunction, S <: Union{Real, AbstractArray}, V <: Union{Real, AbstractArray}, R <: RealOrComplex}(x::AbstractArray{R})
  return g.f((g.a).*x .+ g.b)
end

function prox!{T <: RealOrComplex, R <: Real}(g::Precomposition, x::AbstractArray{T}, y::AbstractArray{T}, gamma::R=1.0)
  y .= g.a .* x .+ g.b
  v = prox!(g.f, y, y, (g.a .* g.a) .* gamma)
  y .-= g.b
  y ./= g.a
  return v
end

function prox_naive{T <: RealOrComplex, R <: Real}(g::Precomposition, x::AbstractArray{T}, gamma::R=1.0)
  z = g.a .* x + g.b
  y, fy = prox_naive(g.f, z, (g.a .* g.a) * gamma)
  return (y - g.b)./g.a, fy
end

fun_name(f::Precomposition) = string("Precomposition by affine diagonal mapping of ", fun_name(f.f))
fun_dom(f::Precomposition) = fun_dom(f.f)
fun_expr(f::Precomposition) = "x ↦ f(diag(a)*x + b)"
fun_params(f::Precomposition) = string("f(x) = ", fun_expr(f.f), ", a = ", length(f.a) == 1 ? string(f.a[1]) : string(typeof(f.a)), ", b = ", length(f.b) == 1 ? string(f.b[1]) : string(typeof(f.b)))
