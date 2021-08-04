# Huber loss function

using LinearAlgebra

export HuberLoss

"""
**Huber loss**

    HuberLoss(ρ=1, μ=1)

Returns the function
```math
f(x) = \\begin{cases}
    \\tfrac{μ}{2}\\|x\\|^2 & \\text{if}\\ \\|x\\| ⩽ ρ \\\\
    ρμ(\\|x\\| - \\tfrac{ρ}{2}) & \\text{otherwise},
\\end{cases}
```
where `ρ` and `μ` are positive parameters.
"""
struct HuberLoss{R, S}
    rho::R
    mu::S
    function HuberLoss{R, S}(rho::R, mu::S) where {R, S}
        if rho <= 0 || mu <= 0
            error("parameters rho and mu must be positive")
        else
            new(rho, mu)
        end
    end
end

is_convex(f::HuberLoss) = true
is_smooth(f::HuberLoss) = true

HuberLoss(rho::R=1, mu::S=1) where {R, S} = HuberLoss{R, S}(rho, mu)

function (f::HuberLoss)(x)
    R = real(eltype(x))
    normx = norm(x)
    if normx <= f.rho
        return f.mu / R(2) * normx^2
    else
        return f.rho * f.mu * (normx - f.rho / R(2))
    end
end

function gradient!(y, f::HuberLoss, x)
    R = real(eltype(x))
    normx = norm(x)
    if normx <= f.rho
        y .= f.mu .* x
        v = f.mu / R(2) * normx^2
    else
        y .= (f.mu * f.rho) / normx .* x
        v = f.rho * f.mu * (normx - f.rho / R(2))
    end
    return v
end

function prox!(y, f::HuberLoss, x, gamma)
    R = real(eltype(x))
    normx = norm(x)
    mugam = f.mu*gamma
    scal = (1 - min(mugam / (1 + mugam), mugam * f.rho / normx))
    for k in eachindex(y)
        y[k] = scal*x[k]
    end
    normy = scal*normx
    if normy <= f.rho
        return f.mu / R(2) * normy^2
    else
        return f.rho * f.mu * (normy - f.rho / R(2))
    end
end

function prox_naive(f::HuberLoss, x, gamma)
    R = real(eltype(x))
    y = (1 - min(f.mu * gamma / (1 + f.mu * gamma), f.mu * gamma * f.rho / norm(x))) * x
    if norm(y) <= f.rho
        return y, f.mu / R(2) * norm(y)^2
    else
        return y, f.rho * f.mu * (norm(y) - f.rho / R(2))
    end
end
