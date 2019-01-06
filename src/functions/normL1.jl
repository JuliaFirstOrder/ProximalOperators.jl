# L1 norm (times a constant, or weighted)

export NormL1

"""
**``L_1`` norm**

	NormL1(λ=1.0)

With a nonnegative scalar parameter λ, returns the function
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, returns the function
```math
f(x) = ∑_i λ_i|x_i|.
```
"""
struct NormL1{T <: Union{Real, AbstractArray}} <: ProximableFunction
	lambda::T
	function NormL1{T}(lambda::T) where {T <: Union{Real, AbstractArray}}
		if !(eltype(lambda) <: Real)
			error("λ must be real")
		end
		if any(lambda .< 0)
			error("λ must be nonnegative")
		else
			new(lambda)
		end
	end
end

is_separable(f::NormL1) = true
is_convex(f::NormL1) = true

"""
	NormL1(λ::Real=1.0)

Returns the function `g(x) = λ||x||_1`, for a real parameter `λ ⩾ 0`.
"""
NormL1(lambda::R=1.0) where {R <: Real} = NormL1{R}(lambda)

"""
	NormL1(λ::Array{Real})

Returns the function `g(x) = sum(λ_i|x_i|, i = 1,...,n)`, for a vector of real
parameters `λ_i ⩾ 0`.
"""
NormL1(lambda::A) where {A <: AbstractArray} = NormL1{A}(lambda)

function (f::NormL1{R})(x::AbstractArray) where R <: Real
	return f.lambda*norm(x, 1)
end

function (f::NormL1{A})(x::AbstractArray) where A <: AbstractArray
	return norm(f.lambda .* x, 1)
end

function prox!(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma::Real=1.0) where {A <: AbstractArray, R <: Real}
	@assert length(y) == length(x) == length(f.lambda)
	fy = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma*f.lambda[i]
		y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
	end
	return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{Complex{R}}, f::NormL1{A}, x::AbstractArray{Complex{R}}, gamma::Real=1.0) where {A <: AbstractArray, R <: Real}
	@assert length(y) == length(x) == length(f.lambda)
	fy = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma*f.lambda[i]
		y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
	end
	return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma::Real=1.0) where {T <: Real, R <: Real}
	@assert length(y) == length(x)
	n1y = R(0)
	gl = gamma*f.lambda
	@inbounds @simd for i in eachindex(x)
		y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
		n1y += y[i] > 0 ? y[i] : -y[i]
	end
	return f.lambda*n1y
end

function prox!(y::AbstractArray{Complex{R}}, f::NormL1{T}, x::AbstractArray{Complex{R}}, gamma::Real=1.0) where {T <: Real, R <: Real}
	@assert length(y) == length(x)
	gl = gamma*f.lambda
	n1y = R(0)
	@inbounds @simd for i in eachindex(x)
		y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
		n1y += abs(y[i])
	end
	return f.lambda*n1y
end

function prox!(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma::AbstractArray) where {A <: AbstractArray, R <: Real}
	@assert length(y) == length(x) == length(f.lambda) == length(gamma)
	fy = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma[i]*f.lambda[i]
		y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
	end
	return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{Complex{R}}, f::NormL1{A}, x::AbstractArray{Complex{R}}, gamma::AbstractArray) where {A <: AbstractArray, R <: Real}
	@assert length(y) == length(x) == length(f.lambda) == length(gamma)
	fy = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma[i]*f.lambda[i]
		y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
	end
	return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma::AbstractArray) where {T <: Real, R <: Real}
	@assert length(y) == length(x) == length(gamma)
	n1y = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma[i]*f.lambda
		y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
		n1y += y[i] > 0 ? y[i] : -y[i]
	end
	return f.lambda*n1y
end

function prox!(y::AbstractArray{Complex{R}}, f::NormL1{T}, x::AbstractArray{Complex{R}}, gamma::AbstractArray) where {T <: Real, R <: Real}
	@assert length(y) == length(x) == length(gamma)
	n1y = R(0)
	@inbounds @simd for i in eachindex(x)
		gl = gamma[i]*f.lambda
		y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
		n1y += abs(y[i])
	end
	return f.lambda*n1y
end

function gradient!(y::AbstractArray{T}, f::NormL1, x::AbstractArray{T}) where T <: Union{Real, Complex}
	y .= f.lambda.*sign.(x)
	return f(x)
end

fun_name(f::NormL1) = "weighted L1 norm"
fun_dom(f::NormL1) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::NormL1{R}) where {R <: Real} = "x ↦ λ||x||_1"
fun_expr(f::NormL1{A}) where {A <: AbstractArray} = "x ↦ sum( λ_i |x_i| )"
fun_params(f::NormL1{R}) where {R <: Real} = "λ = $(f.lambda)"
fun_params(f::NormL1{A}) where {A <: AbstractArray} = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive(f::NormL1, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0) where T <: RealOrComplex
	y = sign.(x).*max.(0.0, abs.(x) .- gamma .* f.lambda)
	return y, norm(f.lambda .* y,1)
end
