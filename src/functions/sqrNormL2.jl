# squared L2 norm (times a constant, or weighted)

export SqrNormL2

"""
**Squared Euclidean norm (weighted)**

    SqrNormL2(λ=1)

With a nonnegative scalar `λ`, returns the function
```math
f(x) = \\tfrac{λ}{2}\\|x\\|^2.
```
With a nonnegative array `λ`, returns the function
```math
f(x) = \\tfrac{1}{2}∑_i λ_i x_i^2.
```
"""
struct SqrNormL2{T <: Union{Real, AbstractArray}}
    lambda::T
    function SqrNormL2{T}(lambda::T) where {T <: Union{Real,AbstractArray}}
        if any(lambda .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::SqrNormL2) = true
is_smooth(f::SqrNormL2) = true
is_separable(f::SqrNormL2) = true
is_quadratic(f::SqrNormL2) = true
is_strongly_convex(f::SqrNormL2) = all(f.lambda .> 0)

SqrNormL2(lambda::T=1) where {T <: Real} = SqrNormL2{T}(lambda)

SqrNormL2(lambda::T) where {T <: AbstractArray} = SqrNormL2{T}(lambda)

function (f::SqrNormL2{S})(x::AbstractArray{T}) where {S <: Real, R <: Real, T <: RealOrComplex{R}}
    return f.lambda / R(2) * norm(x)^2
end

function (f::SqrNormL2{S})(x::AbstractArray{T}) where {S <: AbstractArray, R <: Real, T <: RealOrComplex{R}}
    sqnorm = R(0)
    for k in eachindex(x)
        sqnorm += f.lambda[k] * abs2(x[k])
    end
    return sqnorm / R(2)
end

function gradient!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}) where {S <: Real, R <: Real, T <: RealOrComplex{R}}
    sqnx = R(0)
    for k in eachindex(x)
        y[k] = f.lambda * x[k]
        sqnx += abs2(x[k])
    end
    return f.lambda / R(2) * sqnx
end

function gradient!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}) where {S <: AbstractArray, R <: Real, T <: RealOrComplex{R}}
    sqnx = R(0)
    for k in eachindex(x)
        y[k] = f.lambda[k] * x[k]
        sqnx += f.lambda[k] * abs2(x[k])
    end
    return sqnx / R(2)
end

function prox!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}, gamma::Number) where {S <: Real, R <: Real, T <: RealOrComplex{R}}
    gl = gamma * f.lambda
    sqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gl)
        sqny += abs2(y[k])
    end
    return f.lambda / R(2) * sqny
end

function prox!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}, gamma::Number) where {S <: AbstractArray, R <: Real, T <: RealOrComplex{R}}
    wsqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma * f.lambda[k])
        wsqny += f.lambda[k] * abs2(y[k])
    end
    return wsqny / R(2)
end

function prox!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}, gamma::AbstractArray{R}) where {S <: Real, R <: Real, T <: RealOrComplex{R}}
    sqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma[k] * f.lambda)
        sqny += abs2(y[k])
    end
    return f.lambda / R(2) * sqny
end

function prox!(y::AbstractArray{T}, f::SqrNormL2{S}, x::AbstractArray{T}, gamma::AbstractArray{R}) where {S <: AbstractArray, R <: Real, T <: RealOrComplex{R}}
    wsqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma[k] * f.lambda[k])
        wsqny += f.lambda[k] * abs2(y[k])
    end
    return wsqny / R(2)
end

function prox_naive(f::SqrNormL2, x::AbstractArray{T}, gamma) where {R <: Real, T <: RealOrComplex{R}}
    y = x./(R(1) .+ f.lambda .* gamma)
    return y, real(dot(f.lambda .* y, y)) / R(2)
end
