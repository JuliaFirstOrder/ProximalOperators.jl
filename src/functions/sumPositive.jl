# Sum of the positive components

"""
  SumPositive()

Returns the function `g(x) = sum(max(0, x))`.
"""

immutable SumPositive <: ProximableFunction end

function (f::SumPositive){T <: Real}(x::AbstractArray{T})
  return sum(max.(0.0, x))
end

function prox!{T <: Real}(f::SumPositive, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  fsum = 0.0
  for i in eachindex(x)
    y[i] = x[i] < gamma ? (x[i] > 0.0 ? 0.0 : x[i]) : x[i]-gamma
    fsum += y[i] > 0.0 ? y[i] : 0.0
  end
  return fsum
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

function prox!{T <: Real}(f::SumPositive, x::AbstractArray{T}, y::AbstractArray{T}, gamma::AbstractArray{T})
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
