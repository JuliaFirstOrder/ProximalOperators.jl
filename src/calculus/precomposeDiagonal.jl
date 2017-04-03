# Precompose with diagonal scaling and translation

"""
  PrecomposeDiagonal(f::ProximableConvex, a::AbstractArray, b::AbstractArray)

Returns the function `g(x) = f(a.*x + b)`. Function `f` must be separable, or `a` must be a scalar, for the `prox` of `g` to be computable.
"""

immutable PrecomposeDiagonal{T <: ProximableConvex, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableConvex
  f::T
  a::R
  b::S
  function PrecomposeDiagonal(f::T, a::R, b::S)
    if !(eltype(a) <: Real)
      error("a must have real elements")
    end
    if any(a == 0.0)
      error("elements of a must be nonzero")
    else
      new(f, a, b)
    end
  end
end

is_separable(f::PrecomposeDiagonal) = is_separable(f.f)
is_prox_accurate(f::PrecomposeDiagonal) = is_prox_accurate(f.f)

PrecomposeDiagonal{T <: ProximableConvex, S <: Real}(f::T, a::S=one(S), b::S=zero(S)) = PrecomposeDiagonal{T, S, S}(f, a, b)

PrecomposeDiagonal{T <: ProximableConvex, R <: AbstractArray, S <: Real}(f::T, a::R, b::S=zero(S)) = PrecomposeDiagonal{T, R, S}(f, a, b)

PrecomposeDiagonal{T <: ProximableConvex, R <: Union{AbstractArray, Real}, S <: AbstractArray}(f::T, a::R, b::S) = PrecomposeDiagonal{T, R, S}(f, a, b)

function (g::PrecomposeDiagonal){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f((g.a).*x .+ g.b)
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::PrecomposeDiagonal, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  z = g.a .* x .+ g.b
  v = prox!(y, g.f, z, (g.a .* g.a) .* gamma)
  y .-= g.b
  y ./= g.a
  return v
end

function prox_naive{T <: RealOrComplex}(g::PrecomposeDiagonal, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  z = g.a .* x + g.b
  y, fy = prox_naive(g.f, z, (g.a .* g.a) .* gamma)
  return (y - g.b)./g.a, fy
end

fun_name(f::PrecomposeDiagonal) = string("Precomposition by affine diagonal mapping of ", fun_name(f.f))
fun_dom(f::PrecomposeDiagonal) = fun_dom(f.f)
fun_expr(f::PrecomposeDiagonal) = "x ↦ f(diag(a)*x + b)"
fun_params(f::PrecomposeDiagonal) = string("f(x) = ", fun_expr(f.f), ", a = ", length(f.a) == 1 ? string(f.a[1]) : string(typeof(f.a)), ", b = ", length(f.b) == 1 ? string(f.b[1]) : string(typeof(f.b)))
