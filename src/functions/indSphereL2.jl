# indicator of the L2 norm sphere with given radius

export IndSphereL2

"""
    IndSphereL2(r=1)

Return the indicator function of the Euclidean sphere
```math
S = \\{ x : \\|x\\| = r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.
"""
struct IndSphereL2{R}
    r::R
    function IndSphereL2{R}(r::R) where R
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_set(f::Type{<:IndSphereL2}) = true

IndSphereL2(r::R=1) where R = IndSphereL2{R}(r)

function (f::IndSphereL2)(x)
    R = real(eltype(x))
    if isapprox(norm(x), f.r, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndSphereL2, x, gamma)
    R = real(eltype(x))
    normx = norm(x)
    if normx > 0 # zero-zero?
        scal = f.r/normx
        for k in eachindex(x)
            y[k] = scal*x[k]
        end
    else
        normy = R(0)
        for k in eachindex(x)
            y[k] = randn()
            normy += y[k]*y[k]
        end
        normy = sqrt(normy)
        y .*= f.r/normy
    end
    return R(0)
end

function prox_naive(f::IndSphereL2, x, gamma)
    normx = norm(x)
    if normx > 0
        y = x*f.r/normx
    else
        y = randn(size(x))
        y *= f.r/norm(y)
    end
    return y, real(eltype(x))(0)
end
