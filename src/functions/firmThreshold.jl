
immutable FirmThreshold{T <: Union{Real, AbstractArray}} <: ProximableFunction
  lambda::T
  tau::T
  function FirmThreshold(lambda::T, tau::T )

	  if !(eltype(lambda) <: Real)
	          error("λ must be real")
	  end
	  if !(eltype(tau) <: Real)
	          error("τ must be real")
	  end
	  if any(lambda .< 0)
	          error("λ must be nonnegative")
	  end
	  if any(tau .< 0) || any(tau .> 1)
		  error("τ must be 0 < τ < 1")
	  end
	  if length(tau) != length(lambda)
	          error("λ and τ must have the same length")
	  end
	  new(lambda,tau)
  end
end

"""
  FirmThreshold(λ::Real=1.0,τ::Real=0.7)

  Returns the function `g(x) = sum( τ(λ|x_i|-x_i^2/2) for |x_i|<λ; τλ^2/2 elsewhere)`, for a real parameter `λ ⩾ 0` and `0 < τ < 1`.
"""
FirmThreshold{R <: Real}(lambda::R=1.0,tau::R=0.7) = FirmThreshold{R}(lambda,tau)

FirmThreshold{A <: AbstractArray}(lambda::A,tau::A) = FirmThreshold{A}(lambda,tau)

function (f::FirmThreshold{R}){R <: Real}(x::AbstractArray)
  fy = 0.
  for i in eachindex(x)
	  fy += abs(x[i]) < f.lambda ? f.tau*(f.lambda*abs(x[i]) -0.5*real(dot(x[i],x[i]))) : 0.5*f.tau*f.lambda^2
  end
  return fy
end

function (f::FirmThreshold{A}){A <: AbstractArray}(x::AbstractArray)
  fy = 0.
  for i in eachindex(x)
	  fy += abs(x[i]) < f.lambda[i] ? f.tau[i]*(f.lambda[i]*abs(x[i]) - 0.5*real(dot(x[i],x[i]))) : 0.5*f.tau[i]*f.lambda[i]^2
  end
  return fy
end

function prox!{A <: AbstractArray, R <: RealOrComplex}(y::AbstractArray{R}, f::FirmThreshold{A}, x::AbstractArray{R}, gamma::Real=1.0)
  fy = zero(R)
  for i in eachindex(x)
	  y[i] = abs(x[i]) < f.lambda[i]*f.tau[i] ? 0 : ( abs(x[i]) >= f.lambda[i] ? x[i] : (x[i]-f.lambda[i]*f.tau[i]*sign(x[i]))/(1-f.tau[i])  )
  end
  return f(y)
end

function prox!{T <: Real, R <: RealOrComplex}(y::AbstractArray{R}, f::FirmThreshold{T}, x::AbstractArray{R}, gamma::Real=1.0)
  fy = zero(R)
  for i in eachindex(x)
	  y[i] = abs(x[i]) < f.lambda*f.tau ? 0 : ( abs(x[i]) >= f.lambda ? x[i] : (x[i]-f.lambda*f.tau*sign(x[i]))/(1-f.tau)  )
  end
  return f(y)
end

fun_name(f::FirmThreshold) = "weighted Firm Threshold"
fun_dom(f::FirmThreshold) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::FirmThreshold) = "x ↦ sum(τ(λ|x_i|-x_i^2/2) for |x_i|<λ; τλ^2/2 elsewhere)"
fun_params{R <: Real}(f::FirmThreshold{R}) = "λ = $(f.lambda)"
fun_params{A <: AbstractArray}(f::FirmThreshold{A}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))


function prox_naive{A<: Real,T <: RealOrComplex}(f::FirmThreshold{A}, x::AbstractArray{T}, gamma::Real=1.0)
  y = similar(x)
  for i in eachindex(x)
	  if abs(x[i]) < f.lambda*f.tau
		  y[i] = 0
	  elseif abs(x[i]) >= f.lambda
		  y[i] = x[i]
	  else
		  y[i] = (x[i]-f.lambda*f.tau*sign(x[i]))/(1-f.tau)
	  end
  end
  fy = 0.
  for i in eachindex(x)
          if abs(y[i]) < f.lambda
        	  fy += f.tau*(f.lambda*abs(y[i])-0.5*real(dot(y[i],y[i])))
          else
        	  fy += 0.5*f.tau*f.lambda^2
          end
  end
  return y, fy
end


function prox_naive{A<: AbstractArray,T <: RealOrComplex}(f::FirmThreshold{A}, x::AbstractArray{T}, gamma::Real=1.0)
  y = similar(x)
  for i in eachindex(x)
	  if abs(x[i]) < f.lambda[i]*f.tau[i]
		  y[i] = 0
	  elseif abs(x[i]) >= f.lambda[i]
		  y[i] = x[i]
	  else
		  y[i] = (x[i]-f.lambda[i]*f.tau[i]*sign(x[i]))/(1-f.tau[i])
	  end
  end
  fy = 0.
  for i in eachindex(x)
	  if abs(y[i]) < f.lambda[i]
		  fy += f.tau[i]*(f.lambda[i]*abs(y[i])-0.5*real(dot(y[i],y[i])))
          else
		  fy += 0.5*f.tau[i]*f.lambda[i]^2
          end
  end
  return y, fy
end
