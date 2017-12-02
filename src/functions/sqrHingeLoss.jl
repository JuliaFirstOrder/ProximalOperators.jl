# Hinge loss function

export SqrHingeLoss

"""
**Squared Hinge loss**

  SqrHingeLoss(y, μ=1.0)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\}^2,
```
where `y` is an array and `μ` is a positive parameter.
"""

struct SqrHingeLoss{R <: Real, T <: AbstractArray{R}} <: ProximableFunction
  y::T
  mu::R
  function SqrHingeLoss{R, T}(y::T, mu::R) where {R <: Real, T <: AbstractArray{R}}
    if mu <= 0
      error("parameter mu must be positive")
    else
      new(y, mu)
    end
  end
end

is_separable(f::SqrHingeLoss) = true
is_convex(f::SqrHingeLoss) = true
is_smooth(f::SqrHingeLoss) = true

SqrHingeLoss(b::T, mu::R=1.0) where {R <: Real, T <: AbstractArray{R}} = SqrHingeLoss{R, T}(b, mu)

(f::SqrHingeLoss){R <: Real, T <: AbstractArray{R}}(x::T) = f.mu*sum(max.(zero(R), (one(R) .- f.y.*x)).^2)

function gradient!(y::AbstractArray{R}, f::SqrHingeLoss{R, T}, x::AbstractArray{R}) where {R <: Real, T}
	sum = zero(R)
	for i in eachindex(x)
		zz = 1-f.y[i]*x[i]
		z = max(zero(R), zz)
		y[i] = z .> 0 ? -2*f.mu*f.y[i]*zz : 0
		sum += z^2
	end
	return f.mu*sum
end

function prox!(z::AbstractArray{R}, f::SqrHingeLoss{R, T}, x::AbstractArray{R}, gamma::R=one(R)) where {R, T}
    v = zero(R)
    for k in eachindex(x)
        if f.y[k]*x[k] >= 1
            z[k] = x[k]
        else
            z[k] = (x[k] + 2*f.mu*gamma*f.y[k])/(1+2*f.mu*gamma*f.y[k]^2)
            v += (1-f.y[k]*z[k])^2
        end
    end
    return f.mu*v
end

fun_name(f::SqrHingeLoss) = "squared hinge loss"
fun_dom(f::SqrHingeLoss) = "AbstractArray{Real}"
fun_expr(f::SqrHingeLoss) = "x ↦ μ * sum( max(0,1-b*x_i)^2, i=1,...,n )"
fun_params(f::SqrHingeLoss) = "b = $(typeof(f.y)), μ = $(f.mu)"

function prox_naive(f::SqrHingeLoss{R, T}, x::AbstractArray{R}, gamma::R=one(R)) where {R, T}
    flag = f.y.*x .<= 1
    z = copy(x)
    z[flag] = (x[flag] .+ 2.*f.mu.*gamma.*f.y[flag])./(1.+2.*f.mu.*gamma.*f.y[flag].^2)
    return z, f.mu*sum(max.(0.0, 1.-f.y.*z).^2)
end
