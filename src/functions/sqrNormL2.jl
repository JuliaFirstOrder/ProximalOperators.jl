# squared L2 norm (times a constant, or weighted)

export SqrNormL2

"""
    SqrNormL2(λ=1)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2}\\|x\\|^2.
```
With a nonnegative array `λ`, return the weighted squared Euclidean norm
```math
f(x) = \\tfrac{1}{2}∑_i λ_i x_i^2.
```
"""
struct SqrNormL2{T,SC}
    lambda::T
    function SqrNormL2{T,SC}(lambda::T) where {T,SC}
        if any(lambda .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:SqrNormL2}) = true
is_smooth(f::Type{<:SqrNormL2}) = true
is_separable(f::Type{<:SqrNormL2}) = true
is_generalized_quadratic(f::Type{<:SqrNormL2}) = true
is_strongly_convex(f::Type{SqrNormL2{T,SC}}) where {T,SC} = SC

SqrNormL2(lambda::T=1) where T = SqrNormL2{T,all(lambda .> 0)}(lambda)

function (f::SqrNormL2{S})(x) where {S <: Real}
    return f.lambda / real(eltype(x))(2) * norm(x)^2
end

function (f::SqrNormL2{<:AbstractArray})(x)
    R = real(eltype(x))
    sqnorm = R(0)
    for k in eachindex(x)
        sqnorm += f.lambda[k] * abs2(x[k])
    end
    return sqnorm / R(2)
end

function gradient!(y, f::SqrNormL2{<:Real}, x)
    R = real(eltype(x))
    sqnx = R(0)
    for k in eachindex(x)
        y[k] = f.lambda * x[k]
        sqnx += abs2(x[k])
    end
    return f.lambda / R(2) * sqnx
end

function gradient!(y, f::SqrNormL2{<:AbstractArray}, x)
    R = real(eltype(x))
    sqnx = R(0)
    for k in eachindex(x)
        y[k] = f.lambda[k] * x[k]
        sqnx += f.lambda[k] * abs2(x[k])
    end
    return sqnx / R(2)
end

function prox!(y, f::SqrNormL2{<:Real}, x, gamma::Number)
    R = real(eltype(x))
    gl = gamma * f.lambda
    sqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gl)
        sqny += abs2(y[k])
    end
    return f.lambda / R(2) * sqny
end

function prox!(y, f::SqrNormL2{<:AbstractArray}, x, gamma::Number)
    R = real(eltype(x))
    wsqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma * f.lambda[k])
        wsqny += f.lambda[k] * abs2(y[k])
    end
    return wsqny / R(2)
end

function prox!(y, f::SqrNormL2{<:Real}, x, gamma::AbstractArray)
    R = real(eltype(x))
    sqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma[k] * f.lambda)
        sqny += abs2(y[k])
    end
    return f.lambda / R(2) * sqny
end

function prox!(y, f::SqrNormL2{<:AbstractArray}, x, gamma::AbstractArray)
    R = real(eltype(x))
    wsqny = R(0)
    for k in eachindex(x)
        y[k] = x[k] / (1 + gamma[k] * f.lambda[k])
        wsqny += f.lambda[k] * abs2(y[k])
    end
    return wsqny / R(2)
end

function prox_naive(f::SqrNormL2, x, gamma)
    R = real(eltype(x))
    y = x./(R(1) .+ f.lambda .* gamma)
    return y, real(dot(f.lambda .* y, y)) / R(2)
end
