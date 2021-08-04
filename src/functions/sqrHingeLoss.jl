# Hinge loss function

export SqrHingeLoss

"""
**Squared Hinge loss**

    SqrHingeLoss(y, μ=1)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\}^2,
```
where `y` is an array and `μ` is a positive parameter.
"""
struct SqrHingeLoss{R <: Real, S <: Real, T <: AbstractArray{S}}
    y::T
    mu::R
    function SqrHingeLoss{R, S, T}(y::T, mu::R) where {R <: Real, S <: Real, T <: AbstractArray{S}}
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

SqrHingeLoss(b::T, mu::R=1) where {R <: Real, S <: Real, T <: AbstractArray{S}} = SqrHingeLoss{R, S, T}(b, mu)

(f::SqrHingeLoss)(x::T) where {R <: Real, T <: AbstractArray{R}} = f.mu*sum(max.(R(0), (R(1) .- f.y.*x)).^2)

function gradient!(y::AbstractArray{R}, f::SqrHingeLoss, x::AbstractArray{R}) where R
    sum = R(0)
    for i in eachindex(x)
        zz = 1 - f.y[i] * x[i]
        z = max(R(0), zz)
        y[i] = z .> 0 ? -2 * f.mu * f.y[i] * zz : 0
        sum += z^2
    end
    return f.mu * sum
end

function prox!(z::AbstractArray{R}, f::SqrHingeLoss, x::AbstractArray{R}, gamma) where R
    v = R(0)
    for k in eachindex(x)
        if f.y[k] * x[k] >= 1
            z[k] = x[k]
        else
            z[k] = (x[k] + 2 * f.mu * gamma * f.y[k]) / (1 + 2 * f.mu * gamma * f.y[k]^2)
            v += (1 - f.y[k] * z[k])^2
        end
    end
    return f.mu * v
end

function prox_naive(f::SqrHingeLoss, x::AbstractArray{R}, gamma) where R
    flag = f.y .* x .<= 1
    z = copy(x)
    z[flag] = (x[flag] .+ 2 .* f.mu .* gamma .* f.y[flag]) ./ (1 + 2 .* f.mu .* gamma .* f.y[flag].^2)
    return z, f.mu * sum(max.(0, 1 .- f.y .* z).^2)
end
