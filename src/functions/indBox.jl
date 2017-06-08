# indicator of a generic box

export IndBox, IndBallLinf

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  lb::T
  ub::S
  function IndBox{T,S}(lb::T, ub::S) where {T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}}
    if !(eltype(lb) <: Real && eltype(ub) <: Real)
      error("`lb` and `ub` must be real")
    end
    if any(lb .> ub)
      error("`lb` and `ub` must satisfy `lb <= ub`")
    else
      new(lb, ub)
    end
  end
end

is_separable(f::IndBox) = true
is_convex(f::IndBox) = true
is_set(f::IndBox) = true
is_cone(f::IndBox) = all((f.lb .== -Inf) .+ (f.ub .== +Inf) .> 0)

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

function (f::IndBox){R <: Real}(x::AbstractArray{R})
  for k in eachindex(x)
    if x[k] < IndBox_lb(f,k) || x[k] > IndBox_ub(f,k)
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: Real}(y::AbstractArray{R}, f::IndBox, x::AbstractArray{R}, gamma::Real=one(R))
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

prox!{R <: Real}(y::AbstractArray{R}, f::IndBox, x::AbstractArray{R}, gamma::AbstractArray) = prox!(y, f, x, one(R))

"""
  IndBallLinf(r::Real=1.0)

Returns the indicator function of an infinity-norm ball, that is function
`g(x) = ind{maximum(abs(x)) ⩽ r}` for `r ⩾ 0`.
"""

IndBallLinf{R <: Real}(r::R=1.0) = IndBox(-r, r)

fun_name(f::IndBox) = "indicator of a box"
fun_dom(f::IndBox) = "AbstractArray{Real}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
  string( "lb = ", typeof(f.lb) <: AbstractArray ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
          "ub = ", typeof(f.ub) <: AbstractArray ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)

function prox_naive{R <: Real}(f::IndBox, x::AbstractArray{R}, gamma::Real=1.0)
  y = min.(f.ub, max.(f.lb, x))
  return y, 0.0
end

prox_naive{R <: Real}(f::IndBox, x::AbstractArray{R}, gamma::AbstractArray) = prox_naive(f, x, 1.0)
