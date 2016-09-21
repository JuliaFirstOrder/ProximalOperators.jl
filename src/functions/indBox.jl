# indicator of a generic box

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox{T <: Union{Real, RealArray}, S <: Union{Real, RealArray}} <: IndicatorConvex
  lb::T
  ub::S
  iter_lb
  iter_ub
  IndBox(lb, ub, iter_lb, iter_ub) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub, iter_lb, iter_ub)
end

IndBox(lb::Real, ub::Real) = IndBox{Real, Real}(lb, ub, [1], [1])

IndBox(lb::RealArray, ub::Real) = IndBox{RealArray, Real}(lb, ub, eachindex(lb), [1])

IndBox(lb::Real, ub::RealArray) = IndBox{Real, RealArray}(lb, ub, [1], eachindex(ub))

IndBox(lb::RealArray, ub::RealArray) =
  size(lb) != size(ub) ? error("bounds must have the same dimensions, or at least one of them be scalar") :
  IndBox{RealArray, RealArray}(lb, ub, eachindex(lb), eachindex(ub))

@compat function (f::IndBox{Real,Real})(x::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb || x[k] > f.ub
      return +Inf
    end
  end
  return 0.0
end

@compat function (f::IndBox{RealArray,Real})(x::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb[k] || x[k] > f.ub
      return +Inf
    end
  end
  return 0.0
end

@compat function (f::IndBox{Real,RealArray})(x::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb || x[k] > f.ub[k]
      return +Inf
    end
  end
  return 0.0
end

@compat function (f::IndBox{RealArray,RealArray})(x::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb[k] || x[k] > f.ub[k]
      return +Inf
    end
  end
  return 0.0
end

function prox!(f::IndBox{Real,Real}, x::RealArray, gamma::Real, y::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb
      y[k] = f.lb
    elseif x[k] > f.ub
      y[k] = f.ub
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

function prox!(f::IndBox{RealArray,Real}, x::RealArray, gamma::Real, y::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb[k]
      y[k] = f.lb[k]
    elseif x[k] > f.ub
      y[k] = f.ub
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

function prox!(f::IndBox{Real,RealArray}, x::RealArray, gamma::Real, y::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb
      y[k] = f.lb
    elseif x[k] > f.ub[k]
      y[k] = f.ub[k]
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

function prox!(f::IndBox{RealArray,RealArray}, x::RealArray, gamma::Real, y::RealArray)
  for k in eachindex(x)
    if x[k] < f.lb[k]
      y[k] = f.lb[k]
    elseif x[k] > f.ub[k]
      y[k] = f.ub[k]
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

"""
  IndBallInf(r::Real)

Returns the indicator function of an infinity-norm ball, that is function
`g(x) = ind{maximum(abs(x)) ⩽ r}` for `r ⩾ 0`.
"""

IndBallInf(r::Real) = IndBox(-r, r)

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

function prox_naive(f::IndBox, x::RealArray, gamma::Real=1.0)
  y = min(f.ub, max(f.lb, x))
  return y, 0.0
end
