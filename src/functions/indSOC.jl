# indicator of second-order cones

export IndSOC, IndRotatedSOC

"""
**Indicator of the second-order cone**

    IndSOC()

Returns the indicator of the second-order cone (also known as ice-cream cone or Lorentz cone), that is
```math
C = \\left\\{ (t, x) : \\|x\\| \\leq t \\right\\}.
```
"""

immutable IndSOC <: ProximableFunction end

function (f::IndSOC){T <: Real}(x::AbstractArray{T,1})
  # the tolerance in the following line should be customizable
  if norm(x[2:end]) - x[1] <= 1e-14
    return 0.0
  end
  return +Inf
end

is_convex(f::IndSOC) = true
is_set(f::IndSOC) = true

function prox!{T <: Real}(y::AbstractArray{T,1}, f::IndSOC, x::AbstractArray{T,1}, gamma::T=one(T))
  @views nx = norm(x[2:end])
  t = x[1]
  if t <= -nx
    y[:] = 0.0
  elseif t >= nx
    y[:] = x
  else
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    @views y[2:end] .= r .* x[2:end]
  end
  return 0.0
end

fun_name(f::IndSOC) = "indicator of the second-order cone"
fun_dom(f::IndSOC) = "AbstractArray{Real,1}"
fun_expr(f::IndSOC) = "x ↦ 0 if x[1] >= ||x[2:end]||, +∞ otherwise"
fun_params(f::IndSOC) = "none"

function prox_naive{T <: Real}(f::IndSOC, x::AbstractArray{T,1}, gamma=1.0)
  nx = norm(x[2:end])
  t = x[1]
  if t <= -nx
    y = zeros(x)
  elseif t >= nx
    y = x
  else
    y = zeros(x)
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    y[2:end] .= r .* x[2:end]
  end
  return y, 0.0
end

# ########################
# ROTATED SOC
# ########################

"""
**Indicator of the rotated second-order cone**

    IndRotatedSOC()

Returns the indicator of the *rotated* second-order cone (also known as ice-cream cone or Lorentz cone), that is
```math
C = \\left\\{ (p, q, x) : \\|x\\|^2 \\leq 2\\cdot pq, p \\geq 0, q \\geq 0 \\right\\}.
```
"""

immutable IndRotatedSOC <: ProximableFunction end

function (f::IndRotatedSOC){T <: Real}(x::AbstractArray{T,1})
  if x[1] >= -1e-14 && x[2] >= -1e-14 && norm(x[3:end])^2 - 2*x[1]*x[2] <= 1e-14
    return 0.0
  end
  return +Inf
end

is_convex(f::IndRotatedSOC) = true
is_set(f::IndRotatedSOC) = true

function prox!{T <: Real}(y::AbstractArray{T,1}, f::IndRotatedSOC, x::AbstractArray{T,1}, gamma::T=one(T))
  # sin(pi/4) = cos(pi/4) = 0.7071067811865475
  # rotate x ccw by pi/4
  x1 = 0.7071067811865475*x[1] + 0.7071067811865475*x[2]
  x2 = 0.7071067811865475*x[1] - 0.7071067811865475*x[2]
  # project rotated x onto SOC
  @views nx = sqrt(x2^2+norm(x[3:end])^2)
  t = x1
  if t <= -nx
    y[:] = 0.0
  elseif t >= nx
    y[1] = x1
    y[2] = x2
    @views y[3:end] .= x[3:end]
  else
    r = 0.5 * (1 + t / nx)
    y[1] = r * nx
    y[2] = r * x2
    @views y[3:end] = r .* x[3:end]
  end
  # rotate back y cw by pi/4
  y1 = 0.7071067811865475*y[1] + 0.7071067811865475*y[2]
  y2 = 0.7071067811865475*y[1] - 0.7071067811865475*y[2]
  y[1] = y1
  y[2] = y2
  return 0.0
end

fun_name(f::IndRotatedSOC) = "indicator of the rotated second-order cone"
fun_dom(f::IndRotatedSOC) = "AbstractArray{Real,1}"
fun_expr(f::IndRotatedSOC) = "x ↦ 0 if x[1] ⩾ 0, x[2] ⩾ 0, norm(x[3:end])² ⩽ 2*x[1]*x[2], +∞ otherwise"
fun_params(f::IndRotatedSOC) = "none"

function prox_naive{T <: Real}(f::IndRotatedSOC, x::AbstractArray{T,1}, gamma=1.0)
  g = IndSOC()
  z = copy(x)
  z[1] = 0.7071067811865475*x[1] + 0.7071067811865475*x[2]
  z[2] = 0.7071067811865475*x[1] - 0.7071067811865475*x[2]
  y, = prox_naive(g, z, gamma)
  y1 = 0.7071067811865475*y[1] + 0.7071067811865475*y[2]
  y2 = 0.7071067811865475*y[1] - 0.7071067811865475*y[2]
  y[1] = y1
  y[2] = y2
  return y, 0.0
end
