# indicator of the (primal) exponential cone
# the dual exponential cone is obtained through calculus rules

export IndExpPrimal, IndExpDual

"""
**Indicator of the (primal) exponential cone**

    IndExpPrimal()

Returns the indicator function of the primal exponential cone, that is
```math
C = \\mathrm{cl} \\{ (r,s,t) : s > 0, s⋅e^{r/s} \\leq t \\} \\subset \\mathbb{R}^3.
```
"""
struct IndExpPrimal <: ProximableFunction end

is_convex(f::IndExpPrimal) = true
is_cone(f::IndExpPrimal) = true

"""
**Indicator of the (dual) exponential cone**

    IndExpDual()

Returns the indicator function of the dual exponential cone, that is
```math
C = \\mathrm{cl} \\{ (u,v,w) : u < 0, -u⋅e^{v/u} \\leq w⋅e \\} \\subset \\mathbb{R}^3.
```
"""
IndExpDual() = PrecomposeDiagonal(Conjugate(IndExpPrimal()), -1.0)

EXP_PRIMAL_CALL_TOL = 1e-6
EXP_POLAR_CALL_TOL = 1e-3
EXP_PROJ_TOL = 1e-15
EXP_PROJ_MAXIT = 100

function (f::IndExpPrimal)(x::AbstractArray{R,1}) where R <: Real
  if (x[2] > 0.0 && x[2]*exp(x[1]/x[2]) <= x[3]+EXP_PRIMAL_CALL_TOL) ||
     (x[1] <= EXP_PRIMAL_CALL_TOL && abs(x[2]) <= EXP_PRIMAL_CALL_TOL && x[3] >= -EXP_PRIMAL_CALL_TOL)
    return 0.0
  end
  return +Inf
end

function (f::Conjugate{IndExpPrimal})(x::AbstractArray{R,1}) where R <: Real
  if (x[1] > 0.0 && x[1]*exp(x[2]/x[1]) <= -exp(1)*x[3]+EXP_POLAR_CALL_TOL) ||
     (abs(x[1]) <= EXP_POLAR_CALL_TOL && x[2] <= EXP_POLAR_CALL_TOL && x[3] <= EXP_POLAR_CALL_TOL)
    return 0.0
  end
  return +Inf
end

# Projection onto the cone is performed as in SCS (https://github.com/cvxgrp/scs).
# See the following copyright and permission notices.

# The MIT License (MIT)
#
# Copyright (c) 2012 Brendan O'Donoghue (bodonoghue85@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function prox!(y::AbstractVector{R}, f::IndExpPrimal, x::AbstractVector{R}, gamma::R=1.0) where R <: Real
  r = x[1]
  s = x[2]
  t = x[3]
  if (s*exp(r/s) <= t && s > 0) || (r <= 0 && s == 0 && t >= 0)
    # x in the cone
    y .= x
  elseif (-r < 0 && r*exp(s/r) <= -exp(1)*t) || (-r == 0 && -s >= 0 && -t >= 0)
    # -x in the dual cone (x in the polar cone)
    y .= zero(R)
  elseif r < 0 && s < 0
    # analytical solution
    y[1] = x[1]
    y[2] = max(x[2], 0.0)
    y[3] = max(x[3], 0.0)
  else
    v = x
    ub, lb = getRhoUb(x)
    for iter = 1:EXP_PROJ_MAXIT
      rho = (ub + lb)/2
      g, v = calcGrad(x,rho)
      if g > 0
        lb = rho
      else
        ub = rho
      end
      if ub - lb <= EXP_PROJ_TOL
        break
      end
    end
    y .= v
  end
  return zero(R)
end

function getRhoUb(v)
  lb = 0
  rho = 2.0^(-3)
  g, z = calcGrad(v, rho)
  while g > 0
    lb = rho
    rho = rho*2
    g, z = calcGrad(v, rho)
  end
  ub = rho
  return ub, lb
end

function calcGrad(v, rho)
  x = solve_with_rho(v, rho)
  if x[2] == 0.0
    g = x[1]
  else
    g = x[1] + x[2]*log(x[2]/x[3])
  end
  return g, x
end

function solve_with_rho(v, rho)
  x = zeros(3)
  x[3] = newton_exp_onz(rho, v[2], v[3])
  x[2] = (1/rho)*(x[3] - v[3])*x[3]
  x[1] = v[1] - rho
  return x
end

function newton_exp_onz(rho, y_hat, z_hat)
  t = max(-z_hat,EXP_PROJ_TOL)
  for iter=1:EXP_PROJ_MAXIT
    f = (1.0/rho^2)*t*(t + z_hat) - y_hat/rho + log(t/rho) + 1.0
    fp = (1.0/rho^2)*(2.0*t + z_hat) + 1.0/t
    t = t - f/fp
    if t <= -z_hat
      t = -z_hat
      break
    elseif t <= 0
      t = 0
      break
    elseif abs(f) <= EXP_PROJ_TOL
      break
    end
  end
  z = t + z_hat
  return z
end

fun_name(f::IndExpPrimal) = "indicator exponential cone (primal)"
fun_dom(f::IndExpPrimal) = "AbstractArray{Real}"
fun_expr(f::IndExpPrimal) = "x ↦ 0 if x ∈ cl{(r,s,t) : s > 0, s*exp(r/s) ⩽ t}, +∞ otherwise"
fun_params(f::IndExpPrimal) = "none"

fun_name(f::PrecomposeDiagonal{Conjugate{IndExpPrimal}, R}) where {R <: Real} = "indicator of the exponential cone (dual)"
fun_expr(f::PrecomposeDiagonal{Conjugate{IndExpPrimal}, R}) where {R <: Real} = "x ↦ 0 if x ∈ cl{(u,v,w) : u < 0, -u*exp(v/u) ⩽ w*exp(1)}, +∞ otherwise"
fun_params(f::PrecomposeDiagonal{Conjugate{IndExpPrimal}, R}) where {R <: Real} = "none"

prox_naive(f::IndExpPrimal, x::AbstractArray{R}, gamma::Real=1.0) where {R <: Real} =
  prox(f, x, gamma) # we don't have a much simpler way to do this yet

prox_naive(f::PrecomposeDiagonal{Conjugate{IndExpPrimal}}, x::AbstractArray{R}, gamma::Real=1.0) where {R <: Real} =
  prox(f, x, gamma) # we don't have a much simpler way to do this yet
