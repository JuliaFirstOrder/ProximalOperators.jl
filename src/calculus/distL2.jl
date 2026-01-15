# Euclidean distance from a set

export DistL2

"""
    DistL2(ind_S)

Given `ind_S` the indicator function of a set ``S``, and an optional positive parameter `λ`, return the (weighted) Euclidean distance from ``S``, that is function
```math
g(x) = λ\\mathrm{dist}_S(x) = \\min \\{ λ\\|y - x\\| : y \\in S \\}.
```
"""
struct DistL2{R, T}
    ind::T
    lambda::R
    function DistL2{R, T}(ind::T, lambda::R) where {R, T}
        if !is_set_indicator(ind)
            error("`ind` must be a convex set")
        end
        if lambda <= 0
            error("parameter `λ` must be positive")
        else
            new(ind, lambda)
        end
    end
end

is_proximable(::Type{DistL2{R, T}}) where {R, T} = is_proximable(T)
is_convex(::Type{DistL2{R, T}}) where {R, T} = is_convex(T)

DistL2(ind::T, lambda::R=1) where {R, T} = DistL2{R, T}(ind, lambda)

function (f::DistL2)(x)
    p, = prox(f.ind, x)
    return f.lambda * normdiff(x, p)
end

function prox!(y, f::DistL2, x, gamma)
    R = real(eltype(x))
    prox!(y, f.ind, x)
    d = normdiff(x, y)
    gamlam = gamma * f.lambda
    if gamlam < d
        gamlamd = gamlam/d
        y .= (1 - gamlamd) .*x .+ gamlamd .* y
        return f.lambda * (d - gamlam)
    end
    return R(0)
end

function gradient!(y, f::DistL2, x)
    prox!(y, f.ind, x) # Use y as temporary storage
    dist = normdiff(x, y)
    if dist > 0
        y .= (f.lambda / dist) .* (x .- y)
    else
        y .= 0
    end
    return f.lambda * dist
end

function prox_naive(f::DistL2, x, gamma)
    R = real(eltype(x))
    p, = prox(f.ind, x)
    d = norm(x - p)
    gamlam = gamma * f.lambda
    if d > gamlam
        return x + gamlam/d * (p - x), f.lambda * (d - gamlam)
    end
    return p, R(0)
end
