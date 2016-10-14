# indicator of the (primal) exponential cone

immutable IndExpPrimal <: IndicatorConvex end

immutable IndExpDual <: IndicatorConvex
  g::IndExpPrimal
end

IndExpDual() = IndExpDual(IndExpPrimal())

@compat function (f::IndExpPrimal){R <: Real}(x::AbstractArray{R})
  TOL = min(1e-6, 1e4*eps(R))
  if (x[2] > -TOL && x[2]*exp(x[1]/x[2]) <= x[3]+TOL) || (x[1] <= TOL && abs(x[2]) <= TOL && x[3] >= -TOL)
    return 0.0
  end
  return +Inf
end

@compat function (f::IndExpDual){R <: Real}(x::AbstractArray{R})
  TOL = min(1e-6, 1e4*eps(R))
  if (x[1] < TOL && -x[1]*exp(x[2]/x[1]) <= exp(1)*x[3]+TOL) || (abs(x[1]) <= TOL && x[2] >= -TOL && x[3] >= -TOL)
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
    # minimize ||y - x||^2 subject to se^{r/s} = t
    alpha = 0.001
    beta = 0.5
    y[:] = x
    y[2] = max(1.0, y[2])
    y[3] = max(1.0, y[3])
    l = 1.0
    x_copy = [r; s; t]
    r = (w, z) -> [w - x_copy + z*GradIndExp(w); PenaltyIndExp(w)]
    for iter = 1:100
      KKT = [eye(3)+l*HessIndExp(y) GradIndExp(y); GradIndExp(y)' 0.0 ]
      z = KKT \ -r(y,l)
      dy = z[1:3]
      dl = z[4]
      # backtracking line search
      t = 1.0
      ystep = y + t*dy; lstep = l + t*dl
      while ystep[2] < 0 || (norm(r(ystep, lstep)) > (1 - alpha*t)*norm(r(y, l)))
        t = beta*t
        ystep = y + t*dy
        lstep = l + t*dl
      end
      y[:] = ystep
      l = lstep
      if abs(PenaltyIndExp(y)) < 1e-15 && norm(r(y,l)) <= 1e-15
        break
      end
    end
  end
  return 0.0
end

function prox!{R <: Real}(f::IndExpDual, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
  x_copy = copy(x)
  prox!(f.g, -x, y)
  y[:] += x_copy
  return 0.0
end

function PenaltyIndExp{R <: Real}(w::Array{R})
  return w[2]*exp(w[1]/w[2]) - w[3]
end

function GradIndExp{R <: Real}(w::Array{R})
  r = w[1]/w[2]
  expr = exp(r)
  return [expr; expr*(1 - r); -1]
end

function HessIndExp{R <: Real}(w::Array{R})
    r = w[1]/w[2]
    h = exp(r)*[ 1/w[2] -r/w[2] 0.0; -r/w[2] r^2/w[2] 0.0; 0.0 0.0 0.0 ]
    return h
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
