# Huber loss function

export HuberLoss

"""
**Huber loss**

    HuberLoss(ρ=1.0, μ=1.0)

Returns the function
```math
f(x) = \\begin\{cases\}
  \\tfrac{μ}{2}\\|x\\|^2 & \\text{if}\\ \\|x\\| ⩽ ρ \\\\
  ρμ(\\|x\\| - \\tfrac{ρ}{2}) & \\text{otherwise},
\\end\{cases\}
```
where `ρ` and `μ` are positive parameters.
"""

immutable HuberLoss{R <: Real} <: ProximableFunction
  rho::R
  mu::R
  function HuberLoss{R}(rho::R, mu::R) where {R <: Real}
    if rho <= 0.0 || mu <= 0.0
      error("parameters rho and mu must be positive")
    else
      new(rho, mu)
    end
  end
end

is_convex(f::HuberLoss) = true
is_smooth(f::HuberLoss) = true

HuberLoss{R <: Real}(rho::R=1.0, mu::R=1.0) = HuberLoss{R}(rho, mu)

function (f::HuberLoss){T <: Union{Real, Complex}}(x::AbstractArray{T})
  normx = vecnorm(x)
  if normx <= f.rho
    return (f.mu/2)*normx^2
  else
    return f.rho*f.mu*(normx-f.rho/2)
  end
end

function prox!{T <: Union{Real, Complex}}(y::AbstractArray{T}, f::HuberLoss, x::AbstractArray{T}, gamma::Real=1.0)
  normx = vecnorm(x)
  mugam = f.mu*gamma
  scal = (1-min(mugam/(1+mugam), mugam*f.rho/(normx)))
  for k in eachindex(y)
    y[k] = scal*x[k]
  end
  normy = scal*normx
  if normy <= f.rho
    return (f.mu/2)*normy^2
  else
    return f.rho*f.mu*(normy-f.rho/2)
  end
end

function gradient!{T <: Union{Real, Complex}}(y::AbstractArray{T}, f::HuberLoss, x::AbstractArray{T})
  normx = vecnorm(x)
  if normx <= f.rho
    y .= x
  else
    y .= (f.mu*f.rho)/normx .* x
  end
  return 0.0
end

fun_name(f::HuberLoss) = "Huber loss"
fun_dom(f::HuberLoss) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::HuberLoss) = "x ↦ (μ/2)||x||² if ||x||⩽ρ, μρ(||x||-ρ/2) otherwise"
fun_params(f::HuberLoss) = string("ρ = $(f.rho), μ = $(f.mu)")

function prox_naive{T <: Union{Real, Complex}}(f::HuberLoss, x::AbstractArray{T}, gamma::Real=1.0)
  y = (1-min(f.mu*gamma/(1+f.mu*gamma), f.mu*gamma*f.rho/(vecnorm(x))))*x
  if vecnorm(y) <= f.rho
    return y, (f.mu/2)*vecnorm(y)^2
  else
    return y, f.rho*f.mu*(vecnorm(y)-f.rho/2)
  end
end
