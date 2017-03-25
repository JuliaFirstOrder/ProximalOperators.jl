# Euclidean distance from a set

immutable DistL2{R <: Real} <: ProximableFunction
  ind::IndicatorConvex
  lambda::R
  function DistL2{R}(ind::IndicatorConvex, lambda::R) where {R <: Real}
    if lambda < 0
      error("parameter λ must be nonnegative")
    else
      new(ind, lambda)
    end
  end
end

DistL2{R <: Real}(ind::IndicatorConvex, lambda::R=1.0) = DistL2{R}(ind, lambda)

function (f::DistL2){R <: RealOrComplex}(x::AbstractArray{R})
  p, = prox(f.ind, x)
  return f.lambda*vecnorm(x-p)
end

function prox!{R <: RealOrComplex}(y::AbstractArray{R}, f::DistL2, x::AbstractArray{R}, gamma::Real=1.0)
  p, = prox(f.ind, x)
  d = vecnorm(x-p)
  gamlam = (gamma*f.lambda)
  if gamlam < d
    gamlamd = gamlam/d
    for k in eachindex(p)
      y[k] = (1-gamlamd)*x[k] + gamlamd*p[k]
    end
    return f.lambda*(d-gamlam)
  end
  y[:] = p
  return 0.0
end

fun_name(f::DistL2) = "Euclidean distance from a convex set"
fun_dom(f::DistL2) = fun_dom(f.ind)
fun_expr(f::DistL2) = "x ↦ λ inf { ||x-y|| : y ∈ S} "
fun_params(f::DistL2) = string("λ = $(f.lambda), S = ", typeof(f.ind))

function prox_naive{R <: RealOrComplex}(f::DistL2, x::AbstractArray{R}, gamma::Real=1.0)
  p, = prox(f.ind, x)
  d = vecnorm(x-p)
  gamlam = gamma*f.lambda
  if d > gamlam
    return x + gamlam/d*(p-x), f.lambda*(d-gamlam)
  end
  return p, 0.0
end
