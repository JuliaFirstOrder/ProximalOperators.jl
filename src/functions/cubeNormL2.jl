# cubic L2 norm (times a constant

export CubeNormL2

"""
**Cubic Euclidean norm (weighted)**

    CubeNormL2(λ=1)

With a nonnegative scalar `λ`, returns the function
```math
f(x) = λ\\|x\\|^3.
```
"""
struct CubeNormL2{R <: Real} <: ProximableFunction
    lambda::R
    function CubeNormL2{R}(lambda::R) where R
        if lambda < 0
            error("coefficient λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::CubeNormL2) = true
is_smooth(f::CubeNormL2) = true

CubeNormL2(lambda::R=1) where {R <: Real} = CubeNormL2{R}(lambda)

function (f::CubeNormL2{R})(x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
    return f.lambda * norm(x)^3
end

function gradient!(y::AbstractArray{T}, f::CubeNormL2{R}, x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
    norm_x = norm(x)
    y .= (3 * f.lambda * norm_x) .* x
    return f.lambda * norm_x^3
end

function prox!(y::AbstractArray{T}, f::CubeNormL2{R}, x::AbstractArray{T}, gamma::R=R(1)) where {R, T <: RealOrComplex{R}}
    norm_x = norm(x)
    scale = 2 / (1 + sqrt(1 + 12 * gamma * f.lambda * norm_x))
    y .= scale .* x
    return f.lambda * (scale * norm_x)^3
end

function prox_naive(f::CubeNormL2{R}, x::AbstractArray{T}, gamma=R(1)) where {R, T <: RealOrComplex{R}}
    y = 2 / (1 + sqrt(1 + 12 * gamma * f.lambda * norm(x))) * x
    return y, f.lambda * norm(y)^3
end
