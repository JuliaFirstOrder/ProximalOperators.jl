# indicator of the L1 norm ball with given radius

export IndBallL1

"""
    IndBallL1(r=1.0)

Return the indicator function of the ``L_1`` norm ball
```math
S = \\left\\{ x : \\sum_i |x_i| \\leq r \\right\\}.
```
Parameter `r` must be positive.
"""
struct IndBallL1{R}
    r::R
    function IndBallL1{R}(r::R) where R
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::Type{<:IndBallL1}) = true
is_set_indicator(f::Type{<:IndBallL1}) = true

IndBallL1(r::R=1.0) where R = IndBallL1{R}(r)

function (f::IndBallL1)(x)
    R = real(eltype(x))
    if norm(x, 1) - f.r > f.r*eps(R)
        return R(Inf)
    end
    return R(0)
end

function prox!(y, f::IndBallL1, x::AbstractArray{<:Real}, gamma)
    R = eltype(x)
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    else # do a projection of abs(x) onto simplex then recover signs
        abs_x = abs.(x)
        simplex_proj_condat!(y, f.r, abs_x)
        y .*= sign.(x)
        return R(0)
    end
end

function prox!(y, f::IndBallL1, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    else # do a projection of abs(x) onto simplex then recover signs
        abs_x = real.(abs.(x))
        y_temp = similar(abs_x)
        simplex_proj_condat!(y_temp, f.r, abs_x)
        y .= y_temp .* sign.(x)
        return R(0)
    end
end

function prox_naive(f::IndBallL1, x, gamma)
    R = real(eltype(x))
    # do a simple bisection (aka binary search) on λ
    L = R(0)
    U = maximum(abs, x)
    λ = L
    v = R(0)
    maxit = 120
    for _ in 1:maxit
        λ = (L + U) / 2
        v = sum(max.(abs.(x) .- λ, R(0)))
        # modify lower or upper bound
        (v < f.r) ? U = λ : L = λ
        # exit condition
        if abs(L - U) < (1 + abs(U))*eps(R)
            break
        end
    end
    return sign.(x) .* max.(R(0), abs.(x) .- λ), R(0)
end
