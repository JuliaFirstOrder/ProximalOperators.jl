# logarithmic barrier function

export LogBarrier

"""
**Logarithmic barrier**

    LogBarrier(a=1, b=0, μ=1)

Returns the function
```math
f(x) = -μ⋅∑_i\\log(a⋅x_i+b),
```
for a nonnegative parameter `μ`.
"""
struct LogBarrier{R, S, T}
    a::R
    b::S
    mu::T
    function LogBarrier{R, S, T}(a::R, b::S, mu::T) where {R, S, T}
        if mu <= 0
            error("parameter mu must be positive")
        else
            new(a, b, mu)
        end
    end
end

is_separable(f::Type{<:LogBarrier}) = true
is_convex(f::Type{<:LogBarrier}) = true

LogBarrier(a::R=1, b::S=0, mu::T=1) where {R, S, T} = LogBarrier{R, S, T}(a, b, mu)

function (f::LogBarrier)(x::AbstractArray{T}) where T <: Real
    sumf = T(0)
    v = T(0)
    for i in eachindex(x)
        v = f.a * x[i] + f.b
        if v <= T(0)
            return +Inf
        end
        sumf += log(v)
    end
    return -f.mu * sumf
end

function prox!(y::AbstractArray{T}, f::LogBarrier, x::AbstractArray{T}, gamma::Real=1) where T <: Real
    par = 4 * gamma * f.mu * f.a * f.a
    sumf = T(0)
    z = T(0)
    v = T(0)
    for i in eachindex(x)
        z = f.a * x[i] + f.b
        v = (z + sqrt(z * z + par)) / 2
        y[i] = (v - f.b) / f.a
        sumf += log(v)
    end
    return -f.mu * sumf
end

function prox!(y::AbstractArray{T}, f::LogBarrier, x::AbstractArray{T}, gamma::AbstractArray) where T <: Real
    par = 4 * f.mu * f.a * f.a
    sumf = T(0)
    z = T(0)
    v = T(0)
    for i in eachindex(x)
        par_i = gamma[i] * par
        z = f.a * x[i] + f.b
        v = (z + sqrt(z * z + par_i)) / 2
        y[i] = (v - f.b) / f.a
        sumf += log(v)
    end
    return -f.mu * sumf
end

function gradient!(y::AbstractArray{T}, f::LogBarrier, x::AbstractArray{T}) where T <: Real
    sum = T(0)
    for i in eachindex(x)
        logarg = f.a * x[i] + f.b
        y[i] = -f.mu * f.a / logarg
        sum += log(logarg)
    end
    sum *= -f.mu
    return sum
end

function prox_naive(f::LogBarrier, x::AbstractArray{T,1}, gamma::Union{Real, AbstractArray}=1) where T <: Real
    asqr = f.a * f.a
    z = f.a * x .+ f.b
    y = ((z .+ sqrt.(z .* z .+ 4 * gamma * f.mu * asqr)) / 2 .- f.b) / f.a
    fy = -f.mu * sum(log.(f.a .* y .+ f.b))
    return y, fy
end
