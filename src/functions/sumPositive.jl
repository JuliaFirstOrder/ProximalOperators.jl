# Sum of the positive components

export SumPositive

"""
**Sum of the positive coefficients**

    SumPositive()

Returns the function
```math
f(x) = ∑_i \\max\\{0, x_i\\}.
```
"""

immutable SumPositive <: ProximableFunction end

is_separable(f::SumPositive) = true
is_convex(f::SumPositive) = true

function (f::SumPositive){T <: Real}(x::AbstractArray{T})
  return sum(xi -> max(xi,0), x)
end

function prox!{T <: Real}(y::AbstractArray{T}, f::SumPositive, x::AbstractArray{T}, gamma::Real=1.0)
  fsum = 0.0
  for i in eachindex(x)
    y[i] = x[i] < gamma ? (x[i] > 0.0 ? 0.0 : x[i]) : x[i]-gamma
    fsum += y[i] > 0.0 ? y[i] : 0.0
  end
  return fsum
end

function gradient!{T <: Real}(y::AbstractArray{T}, f::SumPositive, x::AbstractArray{T})
  y .= max.(0, sign.(x))
  return sum(xi -> max(xi,0), x)
end

fun_name(f::SumPositive) = "Sum of the positive coefficients"
fun_dom(f::SumPositive) = "AbstractArray{Real}"
fun_expr(f::SumPositive) = "x ↦ sum(max(0, x))"

function prox_naive{T <: Real}(f::SumPositive, x::AbstractArray{T}, gamma::Real=1.0)
  y = copy(x)
  indpos = x .> 0.0
  y[indpos] = max.(0.0, x[indpos]-gamma)
  return y, sum(max.(0.0, y))
end

# ######################### #
# Prox with multiple gammas #
# ######################### #

function prox!{T <: Real}(y::AbstractArray{T}, f::SumPositive, x::AbstractArray{T}, gamma::AbstractArray{T})
  fsum = 0.0
  for i in eachindex(x)
    y[i] = x[i] < gamma[i] ? (x[i] > 0.0 ? 0.0 : x[i]) : x[i]-gamma[i]
    fsum += y[i] > 0.0 ? y[i] : 0.0
  end
  return fsum
end

function prox_naive{T <: Real}(f::SumPositive, x::AbstractArray{T}, gamma::AbstractArray{T})
  y = copy(x)
  indpos = x .> 0.0
  y[indpos] = max.(0.0, x[indpos]-gamma[indpos])
  return y, sum(max.(0.0, y))
end
