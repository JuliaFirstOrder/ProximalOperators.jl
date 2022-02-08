# L2 norm (times a constant)

export NormL2

"""
    NormL2(λ=1)

With a nonnegative scalar parameter λ, return the ``L_2`` norm
```math
f(x) = λ\\cdot\\sqrt{x_1^2 + … + x_n^2}.
```
"""
struct NormL2{R <: Real}
    lambda::R
    function NormL2{R}(lambda::R) where {R <: Real}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:NormL2}) = true
is_positively_homogeneous(f::Type{<:NormL2}) = true

NormL2(lambda::R=1) where {R <: Real} = NormL2{R}(lambda)

function (f::NormL2)(x::AbstractArray)
    return f.lambda * norm(x)
end

function prox!(y::AbstractArray{T}, f::NormL2, x::AbstractArray{T}, gamma) where T <: RealOrComplex
    normx = norm(x)
    scale = max(0, 1 - f.lambda * gamma / normx)
    for i in eachindex(x)
        y[i] = scale*x[i]
    end
    return f.lambda * scale * normx
end

function gradient!(y::AbstractArray{T}, f::NormL2, x::AbstractArray{T}) where T <: Union{Real, Complex}
    fx = norm(x) # Value of f, without lambda
    if fx == 0
        y .= 0
    else
        y .= (f.lambda / fx) .* x
    end
    return f.lambda * fx
end

function prox_naive(f::NormL2, x::AbstractArray{T}, gamma) where T <: RealOrComplex
    normx = norm(x)
    scale = max(0, 1 -f.lambda * gamma / normx)
    y = scale * x
    return y, f.lambda * scale * normx
end
