# Euclidean distance from a set

export DistL2

"""
**Distance from a convex set**

    DistL2(f)

Given `f` the indicator function of a convex set ``S``, and an optional positive parameter `λ`, returns the (weighted) Euclidean distance from ``S``, that is function
```math
g(x) = \\mathrm{dist}_S(x) = \\min \\{ \\|y - x\\| : y \\in S \\}.
```
"""

immutable DistL2{R <: Real, T <: ProximableFunction} <: ProximableFunction
  ind::T
  lambda::R
  function DistL2{R, T}(ind::T, lambda::R) where {R <: Real, T <: ProximableFunction}
    if !is_set(ind) || !is_convex(ind)
      error("`ind` must be a convex set")
    end
    if lambda < 0
      error("parameter `λ` must be nonnegative")
    else
      new(ind, lambda)
    end
  end
end

is_prox_accurate(f::DistL2) = is_prox_accurate(f.ind)
is_convex(f::DistL2) = is_convex(f.ind)

DistL2{R <: Real, T <: ProximableFunction}(ind::T, lambda::R=1.0) = DistL2{R, T}(ind, lambda)

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
