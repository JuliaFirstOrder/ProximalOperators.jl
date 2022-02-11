# logarithmic barrier function

export LogBarrier

"""
    LogBarrier(a=1, b=0, μ=1)

Return the function
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

function (f::LogBarrier)(x)
    T = eltype(x)
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

function prox!(y, f::LogBarrier, x, gamma)
    T = eltype(x)
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

function prox!(y, f::LogBarrier, x, gamma::AbstractArray)
    T = eltype(x)
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

function gradient!(y, f::LogBarrier, x)
    sumf = eltype(x)(0)
    for i in eachindex(x)
        logarg = f.a * x[i] + f.b
        y[i] = -f.mu * f.a / logarg
        sumf += log(logarg)
    end
    return -f.mu * sumf
end

function prox_naive(f::LogBarrier, x, gamma)
    asqr = f.a * f.a
    z = f.a * x .+ f.b
    y = ((z .+ sqrt.(z .* z .+ 4 * gamma * f.mu * asqr)) / 2 .- f.b) / f.a
    fy = -f.mu * sum(log.(f.a .* y .+ f.b))
    return y, fy
end
