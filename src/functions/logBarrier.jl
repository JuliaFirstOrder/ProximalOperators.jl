# logarithmic barrier function

"""
  LogBarrier(mu::Real=1.0, a::Real=1.0, b::Real=0.0)

Returns the function `g(x) = -mu*sum(log(a*x_i-b), i=1,...,n)`.
"""

immutable LogBarrier{T <: Real} <: ProximableFunction
  mu::T
  a::T
  b::T
  function LogBarrier(mu::T, a::T, b::T)
    if mu <= 0
      error("parameter mu must be positive")
    else
      new(mu, a, b)
    end
  end
end

LogBarrier{T <: Real}(mu::T=1.0, a::T=1.0, b::T=0.0) = LogBarrier{T}(mu, a, b)

@compat function (f::LogBarrier){T <: Real}(x::AbstractArray{T,1})
  sumf = 0.0
  v = 0.0
  for i in eachindex(x)
    v = f.a*x[i]-f.b
    if v <= 0.0
      return +Inf
    end
    sumf += log(v)
  end
  return -f.mu*sumf
end

function prox!{T <: Real}(f::LogBarrier, x::AbstractArray{T,1}, y::AbstractArray{T}, gamma::Real=1.0)
  par = 4*gamma*f.mu*f.a*f.a
  sumf = 0.0
  z = 0.0
  v = 0.0
  for i in eachindex(x)
    z = f.a*x[i] - f.b
    v = (z + sqrt(z*z + par))/2
    y[i] = (v + f.b)/f.a
    sumf += log(v)
  end
  return -f.mu*sumf
end

fun_name(f::LogBarrier) = "logarithmic barrier"
fun_type(f::LogBarrier) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::LogBarrier) = "x ↦ -μ * sum( log(a*x_i-b), i=1,...,n )"
fun_params(f::LogBarrier) = "μ = $(f.mu), a = $(f.a), b = $(f.b)"

function prox_naive{T <: Real}(f::LogBarrier, x::AbstractArray{T,1}, gamma::Real=1.0)
  asqr = f.a*f.a
  z = f.a*x - f.b
  y = ((z + sqrt(z.*z + 4*gamma*f.mu*asqr))/2 + f.b)/f.a
  fy = -f.mu * sum(log(f.a*y-f.b))
  return y, fy
end
