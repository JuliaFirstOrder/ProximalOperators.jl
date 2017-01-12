
# Regularize

immutable Regularize{T <: ProximableFunction, S <: Real, A <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  rho::S
  a::A
  function Regularize(f::T, rho::S, a::A)
    if rho <= 0.0
      error("parameter ρ must be positive")
    else
      new(f, rho, a)
    end
  end
end

Regularize{T <: ProximableFunction, S <: Real, A <: Union{Real, AbstractArray}}(f::T, rho::S, a::A) = Regularize{T, S, A}(f, rho, a)

Regularize{T <: ProximableFunction, S <: Real}(f::T, rho::S=1.0) = Regularize{T, S, S}(f, rho, 0.0)

function (g::Regularize){T <: RealOrComplex}(x::AbstractArray{T})
	return g.f(x) + g.rho/2*vecnorm(x-g.a)^2
end

function prox!{T <: RealOrComplex}(g::Regularize, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  gr= g.rho*gamma
  gr2= 1/(1+gr)
  v = prox!(g.f, gr2.*(x+gr.*g.a), y, gr2*gamma)
  return v + g.rho/2*vecnorm(y-g.a)^2
end

function prox_naive{T <: RealOrComplex}(g::Regularize, x::AbstractArray{T}, gamma::Real=1.0)
  y, v = prox_naive(g.f, x./(1+gamma*g.rho)+g.a./(1/(gamma*g.rho)+1), gamma/(1+gamma*g.rho))
  return y, v + g.rho/2*vecnorm(y-g.a)^2
end

fun_name(f::Regularize) = string("Regularized ", fun_name(f.f))
fun_dom(f::Regularize) = fun_dom(f.f)
fun_expr(f::Regularize) = string(fun_expr(f.f),"+(ρ/2)||x-a||^2")
fun_params(f::Regularize) = "ρ = $(f.rho), λ = $(f.f.lambda), a = $( typeof(f.a)<:Real ? f.a :typeof(f.a) )"
