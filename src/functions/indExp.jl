# indicator of the (primal) exponential cone

immutable IndExpPrimal <: IndicatorConvex end

immutable IndExpDual <: IndicatorConvex
  g::IndExpPrimal
end

IndExpDual() = IndExpDual(IndExpPrimal())

EXP_PRIMAL_CALL_TOL = 1e-6
EXP_DUAL_CALL_TOL = 1e-3
EXP_PROJ_TOL = 1e-15
EXP_PROJ_MAXIT = 100

@compat function (f::IndExpPrimal){R <: Real}(x::AbstractArray{R})
  if (x[2] > 0.0 && x[2]*exp(x[1]/x[2]) <= x[3]+EXP_PRIMAL_CALL_TOL) ||
     (x[1] <= EXP_PRIMAL_CALL_TOL && abs(x[2]) <= EXP_PRIMAL_CALL_TOL && x[3] >= -EXP_PRIMAL_CALL_TOL)
    return 0.0
  end
  return +Inf
end

@compat function (f::IndExpDual){R <: Real}(x::AbstractArray{R})
  if (x[1] < 0.0 && -x[1]*exp(x[2]/x[1]) <= exp(1)*x[3]+EXP_DUAL_CALL_TOL) ||
     (abs(x[1]) <= EXP_DUAL_CALL_TOL && x[2] >= -EXP_DUAL_CALL_TOL && x[3] >= -EXP_DUAL_CALL_TOL)
    return 0.0
  end
  return +Inf
end

function prox!{R <: Real}(f::IndExpPrimal, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
  r = x[1]
  s = x[2]
  t = x[3]
  if (s*exp(r/s) <= t && s > 0) || (r <= 0 && s == 0 && t >= 0)
    # x in the cone
    y[:] = x
  elseif (-r < 0 && r*exp(s/r) <= -exp(1)*t) || (-r == 0 && -s >= 0 && -t >= 0)
    # -x in the dual cone (x in the polar cone)
    y[:] = 0.0
  elseif r < 0 && s < 0
    # analytical solution
    y[1] = x[1]
    y[2] = max(x[2], 0.0)
    y[3] = max(x[3], 0.0)
  else
    # this is the algorithm used in SCS
    # Copyright (c) 2012 Brendan O'Donoghue (bodonoghue85@gmail.com)
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
    y[:] = v
  end
  return 0.0
end

function prox!{R <: Real}(f::IndExpDual, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
  x_copy = copy(x)
  prox!(f.g, -x, y)
  y[:] += x_copy
  return 0.0
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
fun_type(f::IndExpPrimal) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndExpPrimal) = "x ↦ 0 if x ∈ cl{(r,s,t) : s > 0, s*exp(r/s) ⩽ t}"
fun_params(f::IndExpPrimal) = "none"

fun_name(f::IndExpDual) = "indicator of the exponential cone (dual)"
fun_type(f::IndExpDual) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndExpDual) = "x ↦ 0 if x ∈ cl{(u,v,w) : u < 0, -u*exp(v/u) ⩽ w*exp(1)}"
fun_params(f::IndExpDual) = "none"

prox_naive{R <: Real}(f::IndExpPrimal, x::AbstractArray{R}, gamma::Real=1.0) =
  prox(f, x, gamma)

prox_naive{R <: Real}(f::IndExpDual, x::AbstractArray{R}, gamma::Real=1.0) =
  prox(f, x, gamma)
