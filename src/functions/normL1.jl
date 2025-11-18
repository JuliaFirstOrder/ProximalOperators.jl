# L1 norm (times a constant, or weighted)

export NormL1

"""
    NormL1(λ=1)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, return the weighted ``L_1`` norm
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

is_proximable(f::Type{<:NormL1}) = true
is_separable(f::Type{<:NormL1}) = true
is_convex(f::Type{<:NormL1}) = true
is_positively_homogeneous(f::Type{<:NormL1}) = true
is_locally_smooth(f::Type{<:NormL1}) = true

NormL1(lambda::R=1) where R = NormL1{R}(lambda)

(f::NormL1)(x) = f.lambda * norm(x, 1)

(f::NormL1{<:AbstractArray})(x) = norm(f.lambda .* x, 1)

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x)
    n1y = eltype(x)(0)
    gl = gamma * f.lambda
    @inbounds @simd for i in eachindex(x)
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x)
    gl = gamma * f.lambda
    n1y = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x)
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return sum(f.lambda .* abs.(y))
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = eltype(x)(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function gradient!(y, f::NormL1, x)
    y .= f.lambda .* sign.(x)
    return f(x)
end

function prox_naive(f::NormL1, x, gamma)
    y = sign.(x).*max.(0, abs.(x) .- gamma .* f.lambda)
    return y, norm(f.lambda .* y,1)
end
