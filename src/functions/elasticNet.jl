# elastic-net regularization

"""
  ElasticNet(λ::Real=1.0, μ::Real=1.0)

Returns the function `g(x) = μ||x||_1 + (λ/2)||x||²`, for a real parameters `μ, λ ⩾ 0`.
"""

immutable ElasticNet{R <: Real} <: ProximableFunction
  mu::R
  lambda::R
  function ElasticNet{R}(mu::R, lambda::R) where {R <: Real}
    if lambda < 0 || mu < 0
      error("parameters μ, λ must be nonnegative")
    else
      new(mu, lambda)
    end
  end
end

is_separable(f::ElasticNet) = true

ElasticNet{R <: Real}(mu::R=1.0, lambda::R=1.0) = ElasticNet{R}(mu, lambda)

function (f::ElasticNet){R <: RealOrComplex}(x::AbstractArray{R})
  return f.mu*vecnorm(x,1) + (f.lambda/2)*vecnorm(x,2)^2
end

function prox!{R <: Real}(y::AbstractArray{R}, f::ElasticNet{R}, x::AbstractArray{R}, gamma::Real=1.0)
  sqnorm2x = zero(R)
  norm1x = zero(R)
  gm = gamma*f.mu
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
    sqnorm2x += abs2(y[i])
    norm1x += abs(y[i])
  end
  return f.mu*norm1x + (f.lambda/2)*sqnorm2x
end

function prox!{R <: Real}(y::AbstractArray{R}, f::ElasticNet{R}, x::AbstractArray{R}, gamma::AbstractArray{R})
  sqnorm2x = zero(R)
  norm1x = zero(R)
  for i in eachindex(x)
    gm = gamma[i]*f.mu
    gl = gamma[i]*f.lambda
    y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
    sqnorm2x += abs2(y[i])
    norm1x += abs(y[i])
  end
  return f.mu*norm1x + (f.lambda/2)*sqnorm2x
end

function prox!{R <: Real}(y::AbstractArray{Complex{R}}, f::ElasticNet{R}, x::AbstractArray{Complex{R}}, gamma::Real=1.0)
  sqnorm2x = zero(R)
  norm1x = zero(R)
  gm = gamma*f.mu
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = sign(x[i])*max(0, abs(x[i]) - gm)/(1 + gl)
    sqnorm2x += abs2(y[i])
    norm1x += abs(y[i])
  end
  return f.mu*norm1x + (f.lambda/2)*sqnorm2x
end

function prox!{R <: Real}(y::AbstractArray{Complex{R}}, f::ElasticNet{R}, x::AbstractArray{Complex{R}}, gamma::AbstractArray{R})
  sqnorm2x = zero(R)
  norm1x = zero(R)
  for i in eachindex(x)
    gm = gamma[i]*f.mu
    gl = gamma[i]*f.lambda
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

function prox_naive{R <: RealOrComplex}(f::ElasticNet, x::AbstractArray{R}, gamma::Real=1.0)
  uz = max.(0, abs.(x) - gamma*f.mu)/(1 + f.lambda*gamma)
  return sign.(x).*uz, f.mu*vecnorm(uz,1) + (f.lambda/2)*vecnorm(uz)^2
end

function prox_naive{R <: RealOrComplex}(f::ElasticNet, x::AbstractArray{R}, gamma::AbstractArray)
  uz = max.(0, abs.(x) - gamma.*f.mu)./(1 + f.lambda*gamma)
  return sign.(x).*uz, f.mu*vecnorm(uz,1) + (f.lambda/2)*vecnorm(uz)^2
end
