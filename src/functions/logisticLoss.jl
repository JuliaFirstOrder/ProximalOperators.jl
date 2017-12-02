# Logistic loss function

export LogisticLoss

"""
**Logistic loss**

  LogisticLoss(y, μ=1.0)

Returns the function
```math
f(x) = μ⋅∑_i log(1+exp(-y_i⋅x_i))
```
where `y` is an array and `μ` is a positive parameter.
"""

struct LogisticLoss{T <: AbstractArray, R <: Real} <: ProximableFunction
    y::T
    mu::R
    function LogisticLoss{T, R}(y::T, mu::R) where {T, R}
        if mu <= zero(R)
            error("parameter mu must be positive")
        end
        new(y, mu)
    end
end

LogisticLoss(y::T, mu::R=one(R)) where {R, T <: AbstractArray{R}} = LogisticLoss{T, R}(y, mu)

is_separable(f::LogisticLoss) = true
is_convex(f::LogisticLoss) = true
is_prox_accurate(f::LogisticLoss) = false

# f(x)  =  mu log(1 + exp(-y x))

function (f::LogisticLoss{T, R})(x::AbstractArray{R}) where {T, R}
    val = zero(R)
    for k in eachindex(x)
        expyx = exp(f.y[k]*x[k])
        val += log(1.0 + 1.0/expyx)
    end
    return f.mu * val
end

# f'(x) = -mu y exp(-y x) / (1 + exp(-y x))
#       = -mu y / (1 + exp(y x))
#
# Lipschitz constant of gradient: (mu y)

function gradient!(g::AbstractArray{R}, f::LogisticLoss{T, R}, x::AbstractArray{R}) where {T, R}
    val = zero(R)
    for k in eachindex(x)
        expyx = exp(f.y[k]*x[k])
        g[k] = -f.mu * f.y[k] / (1.0 + expyx)
        val += log(1.0 + 1.0/expyx)
    end
    return f.mu * val
end

# Computing proximal operator:
# z = prox(f, x, gamma)
# <==> f'(z) + (z - x)/gamma = 0
# <==> (z - x)/gamma - mu y / (1 + exp(y z)) = 0
# <==> z - x - mu gamma y / (1 + exp(y z)) = 0
#
# Indicating the above condition as F(z) = 0, then
# ==> F'(z) = 1 - (mu gamma y^2 exp(y z))/(1+exp(y z))^2
#
# Newton's method (no damping) to compute z reads:
# z_{k+1} = z_k - F(z_k)/F'(z_k)
#
# To ensure convergence of Newton's method a damping is required.
# The damping coefficient could be computed by backtracking.
#
# Alternatively we can use gradient methods with constant step size.

function prox!(z::AbstractArray{R}, f::LogisticLoss{T, R}, x::AbstractArray{R}, gamma::R=one(R)) where {T, R}
    c = 1.0/gamma # convexity modulus
    L = maximum(abs, f.mu .* f.y) + c # Lipschitz constants
    z .= x
    expyz = similar(z)
    Fz = similar(z)
    F1z = similar(z)
    for k = 1:20
        expyz .= exp.(f.y .* z)
        Fz .= z .- x .- f.mu * gamma * (f.y ./ (1 .+ expyz))
        z .-= Fz ./ L
    end
    expyz .= exp.(f.y .* z)
    val = zero(R)
    for k in eachindex(expyz)
        val += log(1.0 + 1.0/expyz[k])
    end
    return f.mu * val
end
