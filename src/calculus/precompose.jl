# Precompose with diagonal scaling and translation

"""
  Precompose(f::ProximableFunction, a::AbstractArray, b::AbstractArray)

Returns the function `g(x) = f(diag(a)*x) + b`. Function `f` must be separable, or `a` must be a scalar, for the `prox` of `g` to be computable.
"""

immutable Precompose{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  a::R
  b::S
  function Precompose(f::T, a::R, b::S)
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

Precompose{T <: ProximableFunction, S <: Real}(f::T, a::S=one(S), b::S=zero(S)) = Precompose{T, S, S}(f, a, b)

Precompose{T <: ProximableFunction, R <: AbstractArray, S <: Real}(f::T, a::R, b::S=zero(S)) = Precompose{T, R, S}(f, a, b)

Precompose{T <: ProximableFunction, R <: Union{AbstractArray, Real}, S <: AbstractArray}(f::T, a::R, b::S) = Precompose{T, R, S}(f, a, b)

is_prox_accurate(f::Precompose) = is_prox_accurate(f.f)

function (g::Precompose){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f((g.a).*x .+ g.b)
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::Precompose, x::AbstractArray{T}, gamma::Real=1.0)
  y .= g.a .* x .+ g.b
  v = prox!(y, g.f, y, (g.a .* g.a) .* gamma)
  y .-= g.b
  y ./= g.a
  return v
end

function prox_naive{T <: RealOrComplex}(g::Precompose, x::AbstractArray{T}, gamma::Real=1.0)
  z = g.a .* x + g.b
  y, fy = prox_naive(g.f, z, (g.a .* g.a) * gamma)
  return (y - g.b)./g.a, fy
end

fun_name(f::Precompose) = string("Precomposition by affine diagonal mapping of ", fun_name(f.f))
fun_dom(f::Precompose) = fun_dom(f.f)
fun_expr(f::Precompose) = "x ↦ f(diag(a)*x + b)"
fun_params(f::Precompose) = string("f(x) = ", fun_expr(f.f), ", a = ", length(f.a) == 1 ? string(f.a[1]) : string(typeof(f.a)), ", b = ", length(f.b) == 1 ? string(f.b[1]) : string(typeof(f.b)))
