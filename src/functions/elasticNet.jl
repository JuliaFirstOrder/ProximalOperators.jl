# elastic-net regularization

"""
  ElasticNet(λ::Real=1.0, μ::Real=1.0)

Returns the function `g(x) = μ||x||_1 + (λ/2)||x||²`, for a real parameters `μ, λ ⩾ 0`.
"""

immutable ElasticNet{R <: Real} <: ProximableFunction
  mu::R
  lambda::R
  function ElasticNet(mu::R, lambda::R)
    if lambda < 0 || mu < 0
      error("parameters μ, λ must be nonnegative")
    else
      new(mu, lambda)
    end
  end
end

ElasticNet{R <: Real}(mu::R=1.0, lambda::R=1.0) = ElasticNet{R}(mu, lambda)

function (f::ElasticNet){T <: RealOrComplex}(x::AbstractArray{T})
  return f.mu*vecnorm(x,1) + (f.lambda/2)*vecnorm(x,2)^2
end

function prox!{T <: Real}(f::ElasticNet, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  sqnorm2x = zero(Float64)
  norm1x = zero(Float64)
  gm = gamma*f.mu
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
    sqnorm2x += abs2(y[i])
    norm1x += abs(y[i])
  end
  return f.mu*norm1x + (f.lambda/2)*sqnorm2x
end

function prox!{T <: Complex}(f::ElasticNet, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  sqnorm2x = zero(Float64)
  norm1x = zero(Float64)
  gm = gamma*f.mu
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = sign(x[i])*max(0, abs(x[i]) - gm)/(1 + gl)
    sqnorm2x += abs2(y[i])
    norm1x += abs(y[i])
  end
  return f.mu*norm1x + (f.lambda/2)*sqnorm2x
end

fun_name(f::ElasticNet) = "elastic-net regularization"
fun_dom(f::ElasticNet) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::ElasticNet) = "x ↦ μ||x||_1 + (λ/2)||x||²"
fun_params(f::ElasticNet) = "μ = $(f.mu), λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::ElasticNet, x::AbstractArray{T}, gamma::Real=1.0)
  uz = max(0, abs(x) - gamma*f.mu)/(1 + f.lambda*gamma);
  return sign.(x).*uz, f.mu*vecnorm(uz,1) + (f.lambda/2)*vecnorm(uz)^2
end
