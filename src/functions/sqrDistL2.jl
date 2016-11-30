# squared Euclidean distance from a set

immutable SqrDistL2{R <: Real} <: ProximableFunction
  ind::IndicatorConvex
  lambda::R
  function SqrDistL2(ind::IndicatorConvex, lambda::R)
    if lambda < 0
      error("parameter λ must be nonnegative")
    else
      new(ind, lambda)
    end
  end
end

SqrDistL2{R <: Real}(ind::IndicatorConvex, lambda::R=1.0) = SqrDistL2{R}(ind, lambda)

function (f::SqrDistL2){T <: RealOrComplex}(x::AbstractArray{T})
  p, = prox(f.ind, x)
  return (f.lambda/2)*vecnorm(x-p)^2
end

function prox!{T <: RealOrComplex}(f::SqrDistL2, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  p, = prox(f.ind, x)
  sqrd = (f.lambda/2)*vecnorm(x-p)^2
  c1 = 1/(1+f.lambda*gamma)
  c2 = f.lambda*gamma*c1
  for k in eachindex(p)
    y[k] = c1*x[k] + c2*p[k]
  end
  return sqrd*c1^2
end

fun_name(f::SqrDistL2) = "squared Euclidean distance from a convex set"
fun_dom(f::SqrDistL2) = fun_dom(f.ind)
fun_expr(f::SqrDistL2) = "x ↦ (λ/2) inf { ||x-y||^2 : y ∈ S} "
fun_params(f::SqrDistL2) = string("λ = $(f.lambda), S = ", typeof(f.ind))

function prox_naive{T <: RealOrComplex}(f::SqrDistL2, x::AbstractArray{T}, gamma::Real=1.0)
  p, = prox(f.ind, x)
  sqrd = (f.lambda/2)*vecnorm(x-p)^2
  gamlam = f.lambda*gamma
  return 1/(1+gamlam)*x + gamlam/(1+gamlam)*p, sqrd/(1+gamlam)^2
end
