# logarithmic barrier function

immutable LogBarrier{T <: Real} <: ProximableFunction
  a::T
  b::T
  mu::T
  function LogBarrier(a::T, b::T, mu::T)
    if mu <= 0
      error("parameter mu must be positive")
    else
      new(a, b, mu)
    end
  end
end

is_separable(f::LogBarrier) = true

"""
  LogBarrier(a::Real=1.0, b::Real=0.0, mu::Real=1.0)

Returns the function `g(x) = -mu*sum(log(a*x_i+b), i=1,...,n)`.
"""

LogBarrier{T <: Real}(a::T=1.0, b::T=0.0, mu::T=1.0) = LogBarrier{T}(a, b, mu)

function (f::LogBarrier){T <: Real}(x::AbstractArray{T,1})
  sumf = 0.0
  v = 0.0
  for i in eachindex(x)
    v = f.a*x[i]+f.b
    if v <= 0.0
      return +Inf
    end
    sumf += log(v)
  end
  return -f.mu*sumf
end

function prox!{T <: Real}(y::AbstractArray{T}, f::LogBarrier, x::AbstractArray{T,1}, gamma::Real=1.0)
  par = 4*gamma*f.mu*f.a*f.a
  sumf = 0.0
  z = 0.0
  v = 0.0
  for i in eachindex(x)
    z = f.a*x[i] + f.b
    v = (z + sqrt(z*z + par))/2
    y[i] = (v - f.b)/f.a
    sumf += log(v)
  end
  return -f.mu*sumf
end

function prox!{T <: Real}(y::AbstractArray{T}, f::LogBarrier, x::AbstractArray{T,1}, gamma::AbstractArray)
  par = 4*f.mu*f.a*f.a
  sumf = 0.0
  z = 0.0
  v = 0.0
  for i in eachindex(x)
    par_i = gamma[i]*par
    z = f.a*x[i] + f.b
    v = (z + sqrt(z*z + par_i))/2
    y[i] = (v - f.b)/f.a
    sumf += log(v)
  end
  return -f.mu*sumf
end

fun_name(f::LogBarrier) = "logarithmic barrier"
fun_dom(f::LogBarrier) = "AbstractArray{Real}"
fun_expr(f::LogBarrier) = "x ↦ -μ * sum( log(a*x_i+b), i=1,...,n )"
fun_params(f::LogBarrier) = "a = $(f.a), b = $(f.b), μ = $(f.mu)"

function prox_naive{T <: Real}(f::LogBarrier, x::AbstractArray{T,1}, gamma::Union{Real, AbstractArray}=1.0)
  asqr = f.a*f.a
  z = f.a*x + f.b
  y = ((z + sqrt.(z.*z + 4*gamma*f.mu*asqr))/2 - f.b)/f.a
  fy = -f.mu * sum(log.(f.a*y+f.b))
  return y, fy
end
