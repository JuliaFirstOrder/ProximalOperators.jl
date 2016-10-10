# indicator of a generic box

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: IndicatorConvex
  lb::T
  ub::S
  function IndBox(lb::T, ub::S)
    if any(lb .> ub)
      error("arguments lb, ub must satisfy lb <= ub")
    else
      new(lb, ub)
    end
  end
end

IndBox{T <: Real}(lb::T, ub::T) = IndBox{T, T}(lb, ub)

IndBox{T <: AbstractArray, S <: Real}(lb::T, ub::S) = IndBox{T, S}(lb, ub)

IndBox{T <: Real, S <: AbstractArray}(lb::T, ub::S) = IndBox{T, S}(lb, ub)

IndBox{T <: AbstractArray, S <: AbstractArray}(lb::T, ub::S) =
  size(lb) != size(ub) ? error("bounds must have the same dimensions, or at least one of them be scalar") :
  IndBox{T, S}(lb, ub)

IndBox_lb{T <: Real, S}(f::IndBox{T, S}, i) = f.lb
IndBox_lb{T <: AbstractArray, S}(f::IndBox{T, S}, i) = f.lb[i]
IndBox_ub{T, S <: Real}(f::IndBox{T, S}, i) = f.ub
IndBox_ub{T, S <: AbstractArray}(f::IndBox{T, S}, i) = f.ub[i]

@compat function (f::IndBox){R <: Real}(x::AbstractArray{R})
  for k in eachindex(x)
    if x[k] < IndBox_lb(f,k) || x[k] > IndBox_ub(f,k)
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: Real}(f::IndBox, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
  for k in eachindex(x)
    if x[k] < IndBox_lb(f,k)
      y[k] = IndBox_lb(f,k)
    elseif x[k] > IndBox_ub(f,k)
      y[k] = IndBox_ub(f,k)
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

IndBallInf{R <: Real}(r::R) = IndBox(-r, r)

"""
  IndNonnegative()

Returns the indicator function the nonnegative orthant, that is

  `g(x) = 0 if x ⩾ 0, +∞ otherwise`
"""

IndNonnegative() = IndBox(0.0, +Inf)

"""
  IndNonpositive()

Returns the indicator function the nonpositive orthant, that is

  `g(x) = 0 if x ⩽ 0, +∞ otherwise`
"""

IndNonpositive() = IndBox(-Inf, 0.0)

fun_name(f::IndBox) = "indicator of a box"
fun_type(f::IndBox) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
  string( "lb = ", typeof(f.lb) <: AbstractArray ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
          "ub = ", typeof(f.ub) <: AbstractArray ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)

function prox_naive{R <: Real}(f::IndBox, x::AbstractArray{R}, gamma::Real=1.0)
  y = min(f.ub, max(f.lb, x))
  return y, 0.0
end
