# elastic-net regularization

export ElasticNet

"""
**Elastic-net regularization**

    ElasticNet(μ=1, λ=1)

Returns the function
```math
f(x) = μ\\|x\\|_1 + (λ/2)\\|x\\|^2,
```
for nonnegative parameters `μ` and `λ`.
"""
struct ElasticNet{R <: Real, S <: Real} <: ProximableFunction
    mu::R
    lambda::S
    function ElasticNet{R, S}(mu::R, lambda::S) where {R <: Real, S <: Real}
        if lambda < 0 || mu < 0
            error("parameters `μ` and `λ` must be nonnegative")
        else
            new(mu, lambda)
        end
    end
end

is_separable(f::ElasticNet) = true
is_prox_accurate(f::ElasticNet) = true
is_convex(f::ElasticNet) = true

ElasticNet(mu::R=1, lambda::S=1) where {R <: Real, S <: Real} = ElasticNet{R, S}(mu, lambda)

function (f::ElasticNet)(x::AbstractArray{T}) where {
    R <: Real, T <: RealOrComplex{R}
}
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

function prox!(y::AbstractArray{R}, f::ElasticNet, x::AbstractArray{R}, gamma::R=R(1)) where {
    R <: Real,
}
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

function prox!(y::AbstractArray{R}, f::ElasticNet, x::AbstractArray{R}, gamma::AbstractArray{R}) where {
    R <: Real
}
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

function prox!(y::AbstractArray{Complex{R}}, f::ElasticNet, x::AbstractArray{Complex{R}}, gamma::R=R(1)) where {
    R <: Real
}
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

function prox!(y::AbstractArray{Complex{R}}, f::ElasticNet, x::AbstractArray{Complex{R}}, gamma::AbstractArray{R}) where {
    R <: Real
}
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

function gradient!(y::AbstractArray{T}, f::ElasticNet, x::AbstractArray{T}) where {
    R <: Real, T <: RealOrComplex{R}
}
    # Gradient of 1 norm
    y .= f.mu .* sign.(x)
    # Gradient of 2 norm
    y .+= f.lambda .* x
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

fun_name(f::ElasticNet) = "elastic-net regularization"
fun_dom(f::ElasticNet) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::ElasticNet) = "x ↦ μ||x||_1 + (λ/2)||x||²"
fun_params(f::ElasticNet) = "μ = $(f.mu), λ = $(f.lambda)"

function prox_naive(f::ElasticNet, x::AbstractArray{T}, gamma::R=R(1)) where {
    R <: Real, T <: RealOrComplex{R}
}
    uz = max.(0, abs.(x) .- gamma * f.mu)/(1 + f.lambda * gamma)
    return sign.(x) .* uz, f.mu * norm(uz, 1) + f.lambda / R(2) * norm(uz)^2
end

function prox_naive(f::ElasticNet, x::AbstractArray{T}, gamma::AbstractArray{R}) where {
    R <: Real, T <: RealOrComplex{R}
}
    uz = max.(0, abs.(x) .- gamma .* f.mu)./(1 .+ f.lambda .* gamma)
    return sign.(x) .* uz, f.mu * norm(uz, 1) + f.lambda / R(2) * norm(uz)^2
end
