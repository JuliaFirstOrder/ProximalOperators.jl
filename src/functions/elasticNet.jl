# elastic-net regularization

export ElasticNet

"""
    ElasticNet(μ=1, λ=1)

Return the function
```math
f(x) = μ\\|x\\|_1 + (λ/2)\\|x\\|^2,
```
for nonnegative parameters `μ` and `λ`.
"""
struct ElasticNet{R, S}
    mu::R
    lambda::S
    function ElasticNet{R, S}(mu::R, lambda::S) where {R, S}
        if lambda < 0 || mu < 0
            error("parameters `μ` and `λ` must be nonnegative")
        else
            new(mu, lambda)
        end
    end
end

is_separable(f::Type{<:ElasticNet}) = true
is_prox_accurate(f::Type{<:ElasticNet}) = true
is_convex(f::Type{<:ElasticNet}) = true

ElasticNet(mu::R=1, lambda::S=1) where {R, S} = ElasticNet{R, S}(mu, lambda)

function (f::ElasticNet)(x)
    R = real(eltype(x))
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

function prox!(y, f::ElasticNet, x::AbstractArray{R}, gamma) where {R <: Real}
    sqnorm2x = R(0)
    norm1x = R(0)
    gm = gamma * f.mu
    gl = gamma * f.lambda
    for i in eachindex(x)
        y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
        sqnorm2x += abs2(y[i])
        norm1x += abs(y[i])
    end
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x::AbstractArray{R}, gamma::AbstractArray) where {R <: Real}
    sqnorm2x = R(0)
    norm1x = R(0)
    for i in eachindex(x)
        gm = gamma[i] * f.mu
        gl = gamma[i] * f.lambda
        y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
        sqnorm2x += abs2(y[i])
        norm1x += abs(y[i])
    end
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x::AbstractArray{C}, gamma) where {C <: Complex}
    R = real(C)
    sqnorm2x = R(0)
    norm1x = R(0)
    gm = gamma * f.mu
    gl = gamma * f.lambda
    for i in eachindex(x)
        y[i] = sign(x[i]) * max(0, abs(x[i]) - gm)/(1 + gl)
        sqnorm2x += abs2(y[i])
        norm1x += abs(y[i])
    end
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x::AbstractArray{C}, gamma::AbstractArray) where {C <: Complex}
    R = real(C)
    sqnorm2x = R(0)
    norm1x = R(0)
    for i in eachindex(x)
        gm = gamma[i] * f.mu
        gl = gamma[i] * f.lambda
        y[i] = sign(x[i]) * max(0, abs(x[i]) - gm)/(1 + gl)
        sqnorm2x += abs2(y[i])
        norm1x += abs(y[i])
    end
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function gradient!(y, f::ElasticNet, x)
    R = real(eltype(x))
    # Gradient of 1 norm
    y .= f.mu .* sign.(x)
    # Gradient of 2 norm
    y .+= f.lambda .* x
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

function prox_naive(f::ElasticNet, x, gamma)
    R = real(eltype(x))
    uz = max.(0, abs.(x) .- gamma .* f.mu)./(1 .+ f.lambda .* gamma)
    return sign.(x) .* uz, f.mu * norm(uz, 1) + f.lambda / R(2) * norm(uz)^2
end
