# Regularize

export Regularize

"""
  Regularize(f::ProximableFunction, ρ::Real, a::AbstractArray)

Given function `f`, returns `g(x) = f(x) + (ρ/2)||x-a||²`.
"""

immutable Regularize{T <: ProximableFunction, S <: Real, A <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  rho::S
  a::A
  function Regularize{T,S,A}(f::T, rho::S, a::A) where {T <: ProximableFunction, S <: Real, A <: Union{Real, AbstractArray}}
    if rho <= 0.0
      error("parameter `ρ` must be positive")
    else
      new(f, rho, a)
    end
  end
end

is_separable(f::Regularize) = is_separable(f.f)
is_prox_accurate(f::Regularize) = is_prox_accurate(f.f)
is_convex(f::Regularize) = is_convex(f.f)
is_smooth(f::Regularize) = is_smooth(f.f)
is_quadratic(f::Regularize) = is_quadratic(f.f)
is_generalized_quadratic(f::Regularize) = is_generalized_quadratic(f.f)
is_strongly_convex(f::Regularize) = true

Regularize{T <: ProximableFunction, S <: Real, A <: AbstractArray}(f::T, rho::S, a::A) = Regularize{T, S, A}(f, rho, a)

Regularize{T <: ProximableFunction, S <: Real}(f::T, rho::S=one(S), a::S=zero(S)) = Regularize{T, S, S}(f, rho, a)

function (g::Regularize){T <: RealOrComplex}(x::AbstractArray{T})
	return g.f(x) + g.rho/2*vecnorm(x-g.a)^2
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::Regularize, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  gr = g.rho*gamma
  gr2 = 1.0 ./ (1.0+gr)
  v = prox!(y, g.f, gr2.*(x+gr.*g.a), gr2.*gamma)
  return v + g.rho/2*vecnorm(y-g.a)^2
end

fun_name(f::Regularize) = string("Regularized ", fun_name(f.f))
fun_dom(f::Regularize) = fun_dom(f.f)
fun_expr(f::Regularize) = string(fun_expr(f.f), "+(ρ/2)||x-a||²")
fun_params(f::Regularize) = "ρ = $(f.rho), λ = $(f.f.lambda), a = $( typeof(f.a)<:Real ? f.a :typeof(f.a) )"

function prox_naive{T <: RealOrComplex}(g::Regularize, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  y, v = prox_naive(g.f, x./(1+gamma.*g.rho)+g.a./(1.0./(gamma.*g.rho)+1.0), gamma./(1.0+gamma.*g.rho))
  return y, v + g.rho/2*vecnorm(y-g.a)^2
end
