# L1 norm (times a constant, or weighted)

export NormL1

"""
**``L_1`` norm**

    NormL1(λ=1)

With a nonnegative scalar parameter λ, returns the function
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, returns the function
```math
f(x) = ∑_i λ_i|x_i|.
```
"""
struct NormL1{T}
    lambda::T
    function NormL1{T}(lambda::T) where T
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
is_positively_homogeneous(f::NormL1) = true

NormL1(lambda::R=1) where R = NormL1{R}(lambda)

function (f::NormL1{R})(x) where R <: Number
    return f.lambda * norm(x, 1)
end

function (f::NormL1{A})(x) where A <: AbstractArray
    return norm(f.lambda .* x, 1)
end

function prox!(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma) where {A <: AbstractArray, R <: Real}
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{C}, f::NormL1{A}, x::AbstractArray{C}, gamma) where {A <: AbstractArray, C <: Complex}
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma) where {T <: Real, R <: Real}
    @assert length(y) == length(x)
    n1y = R(0)
    gl = gamma * f.lambda
    @inbounds @simd for i in eachindex(x)
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y::AbstractArray{C}, f::NormL1{T}, x::AbstractArray{C}, gamma) where {T <: Real, C <: Complex}
    @assert length(y) == length(x)
    gl = gamma * f.lambda
    n1y = real(C)(0)
    @inbounds @simd for i in eachindex(x)
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function prox!(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma::AbstractArray) where {A <: AbstractArray, R <: Real}
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{C}, f::NormL1{A}, x::AbstractArray{C}, gamma::AbstractArray) where {A <: AbstractArray, C <: Complex}
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma::AbstractArray) where {T <: Real, R <: Real}
    @assert length(y) == length(x) == length(gamma)
    n1y = R(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y::AbstractArray{C}, f::NormL1{T}, x::AbstractArray{C}, gamma::AbstractArray) where {T <: Real, C <: Complex}
    @assert length(y) == length(x) == length(gamma)
    n1y = real(C)(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function gradient!(y::AbstractArray{T}, f::NormL1, x::AbstractArray{T}) where T <: Union{Real, Complex}
    y .= f.lambda .* sign.(x)
    return f(x)
end

function prox_naive(f::NormL1, x, gamma)
    y = sign.(x).*max.(0, abs.(x) .- gamma .* f.lambda)
    return y, norm(f.lambda .* y,1)
end
