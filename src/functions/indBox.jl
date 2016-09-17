# indicator of a generic box

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox <: IndicatorConvex
  lb
  ub
  iter_lb
  iter_ub
  IndBox(lb::Array, ub::Array) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub, eachindex(lb), eachindex(ub))
  IndBox(lb::Array, ub) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub, eachindex(lb), [1])
  IndBox(lb, ub::Array) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub, [1], eachindex(ub))
  IndBox(lb, ub) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub, [1], [1])
end

@compat function (f::IndBox)(x::Array{Float64})
  iter_x = eachindex(x)
  if length(f.iter_lb) == length(iter_x) && length(f.iter_ub) == length(iter_x)
    for k in iter_x
      if x[k] < f.lb[k] || x[k] > f.ub[k] return +Inf end
    end
  elseif length(f.iter_lb) == length(iter_x) && length(f.iter_ub) == 1
    for k in iter_x
      if x[k] < f.lb[k] || x[k] > f.ub return +Inf end
    end
  elseif length(f.iter_ub) == length(iter_x) && length(f.iter_lb) == 1
    for k in iter_x
      if x[k] < f.lb || x[k] > f.ub[k] return +Inf end
    end
  elseif length(f.iter_lb) == 1 && length(f.iter_ub) == 1
    for k in iter_x
      if x[k] < f.lb || x[k] > f.ub return +Inf end
    end
  else
    error("the argument is incompatible with the boundaries")
  end
  return 0.0
end

function prox!(f::IndBox, x::Array{Float64}, gamma::Float64, y::Array{Float64})
  iter_x = eachindex(x)
  if length(f.iter_lb) == length(iter_x) && length(f.iter_ub) == length(iter_x)
    for k in iter_x
      if x[k] < f.lb[k] y[k] = f.lb[k]
      elseif x[k] > f.ub[k] y[k] = f.ub[k]
      else y[k] = x[k] end
    end
  elseif length(f.iter_lb) == length(iter_x) && length(f.iter_ub) == 1
    for k in iter_x
      if x[k] < f.lb[k] y[k] = f.lb[k]
      elseif x[k] > f.ub y[k] = f.ub
      else y[k] = x[k] end
    end
  elseif length(f.iter_ub) == length(iter_x) && length(f.iter_lb) == 1
    for k in iter_x
      if x[k] < f.lb y[k] = f.lb
      elseif x[k] > f.ub[k] y[k] = f.ub[k]
      else y[k] = x[k] end
    end
  elseif length(f.iter_lb) == 1 && length(f.iter_ub) == 1
    for k in iter_x
      if x[k] < f.lb y[k] = f.lb
      elseif x[k] > f.ub y[k] = f.ub
      else y[k] = x[k] end
    end
  else
    error("the argument is incompatible with the boundaries")
  end
  return 0.0
end

"""
  IndBallInf(r::Float64)

Returns the indicator function of an infinity-norm ball, that is function
`g(x) = ind{maximum(abs(x)) ⩽ r}` for `r ⩾ 0`.
"""

IndBallInf(r::Float64) = IndBox(-r, r)

"""
  IndNonnegative()

Returns the indicator function the nonnegative orthant, that is

  `g(x) = 0 if x ⩾ 0, +∞ otherwise`
"""

IndNonnegative() = IndBox(0, +Inf)

"""
  IndNonpositive()

Returns the indicator function the nonpositive orthant, that is

  `g(x) = 0 if x ⩽ 0, +∞ otherwise`
"""

IndNonpositive() = IndBox(-Inf, 0)

fun_name(f::IndBox) = "indicator of a box"
fun_type(f::IndBox) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
  string( "lb = ", typeof(f.lb) <: Array ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
          "ub = ", typeof(f.ub) <: Array ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)

function prox_naive(f::IndBox, x::Array{Float64}, gamma::Float64=1.0)
  y = min(f.ub, max(f.lb, x))
  return y, 0.0
end
