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

LogisticLoss(y::T, mu::R=1.0) where {T, R} = LogisticLoss{T, R}(y, mu)

is_separable(f::LogisticLoss) = true
is_convex(f::LogisticLoss) = true

# f(x)  =  mu log(1 + exp(-y x))

function (f::LogisticLoss{T, R})(x::S) where {S <: AbstractArray, T, R}
    val = zero(R)
    for k in eachindex(x)
        expyx = exp(f.y[k]*x[k])
        val += log(1.0 + 1.0/expyx)
    end
    return f.mu * val
end

# f'(x) = -mu y exp(-y x) / (1 + exp(-y x))
#       = -mu y / (1 + exp(y x))

function gradient!(g::S, f::LogisticLoss{T, R}, x::S) where {S <: AbstractArray, T, R}
    val = zero(R)
    for k in eachindex(x)
        expyx = exp(f.y[k]*x[k])
        g[k] = -f.mu * f.y[k] / (1.0 + expyx)
        val += log(1.0 + 1.0/expyx)
    end
    return f.mu * val
end

# z = prox(f, x, gamma)
# ==> f'(z) + (z - x)/gamma = 0
# ==> -mu y / (1 + exp(y z)) + (z - x)/gamma = 0
# ==> -mu gamma y / (1 + exp(y z)) + z - x = 0

# TODO: implement prox! using Newton method?
