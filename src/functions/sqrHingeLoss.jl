# Hinge loss function

export SqrHingeLoss

"""
    SqrHingeLoss(y, μ=1)

Return the squared Hinge loss
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\}^2,
```
where `y` is an array and `μ` is a positive parameter.
"""
struct SqrHingeLoss{R, T}
    y::T
    mu::R
    function SqrHingeLoss{R, T}(y::T, mu::R) where {R, T}
        if mu <= 0
            error("parameter mu must be positive")
        else
            new(y, mu)
        end
    end
end

is_separable(f::Type{<:SqrHingeLoss}) = true
is_convex(f::Type{<:SqrHingeLoss}) = true
is_smooth(f::Type{<:SqrHingeLoss}) = true

SqrHingeLoss(b::T, mu::R=1) where {R, T} = SqrHingeLoss{R, T}(b, mu)

function (f::SqrHingeLoss)(x)
    R = eltype(x)
    return f.mu * sum(max.(R(0), (R(1) .- f.y .* x)).^2)
end

function gradient!(y, f::SqrHingeLoss, x)
    R = eltype(x)
    sum = R(0)
    for i in eachindex(x)
        zz = 1 - f.y[i] * x[i]
        z = max(R(0), zz)
        y[i] = z .> 0 ? -2 * f.mu * f.y[i] * zz : 0
        sum += z^2
    end
    return f.mu * sum
end

function prox!(z, f::SqrHingeLoss, x, gamma)
    v = eltype(x)(0)
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

function prox_naive(f::SqrHingeLoss, x, gamma)
    flag = f.y .* x .<= 1
    z = copy(x)
    z[flag] = (x[flag] .+ 2 .* f.mu .* gamma .* f.y[flag]) ./ (1 + 2 .* f.mu .* gamma .* f.y[flag].^2)
    return z, f.mu * sum(max.(0, 1 .- f.y .* z).^2)
end
