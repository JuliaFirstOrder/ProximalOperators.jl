# L0 pseudo-norm (times a constant)

export NormL0

"""
**``L_0`` pseudo-norm**

    NormL0(λ=1)

Returns the function
```math
f(x) = λ\\cdot\\mathrm{nnz}(x)
```
for a nonnegative parameter `λ`.
"""
struct NormL0{R <: Real}
    lambda::R
    function NormL0{R}(lambda::R) where {R <: Real}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

NormL0(lambda::R=1) where {R <: Real} = NormL0{R}(lambda)

function (f::NormL0)(x::AbstractArray{T}) where {R, T <: RealOrComplex{R}}
    return f.lambda * R(count(v -> v != 0, x))
end

function prox!(y::AbstractArray{T}, f::NormL0, x::AbstractArray{T}, gamma::Real=1) where {R, T <: RealOrComplex{R}}
    countnzy = R(0)
    gl = gamma * f.lambda
    for i in eachindex(x)
        over = abs(x[i]) > sqrt(2 * gl)
        y[i] = over * x[i]
        countnzy += over
    end
    return f.lambda * countnzy
end

function prox_naive(f::NormL0, x::AbstractArray{T}, gamma::Real=1) where {R, T <: RealOrComplex{R}}
    over = abs.(x) .> sqrt(2 * gamma * f.lambda)
    y = x.*over
    return y, f.lambda * R(count(v -> v != 0, y))
end
