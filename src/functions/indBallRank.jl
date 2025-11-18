# indicator of the ball of matrices with (at most) a given rank

using LinearAlgebra
using TSVD

export IndBallRank

"""
    IndBallRank(r=1)

Return the indicator function of the set of matrices of rank at most `r`:
```math
S = \\{ X : \\mathrm{rank}(X) \\leq r \\},
```
Parameter `r` must be a positive integer.
"""
struct IndBallRank{I}
    r::I
    function IndBallRank{I}(r::I) where {I}
        if r <= 0
            error("parameter r must be a positive integer")
        else
            new(r)
        end
    end
end

is_set_indicator(f::Type{<:IndBallRank}) = true
is_proximable(f::Type{<:IndBallRank}) = false

IndBallRank(r::I=1) where I = IndBallRank{I}(r)

function (f::IndBallRank)(x)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r return R(0) end
    U, S, V = tsvd(x, f.r+1)
    # the tolerance in the following line should be customizable
    if S[end]/S[1] <= 1e-7
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndBallRank, x, gamma)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        y .= x
        return R(0)
    end
    U, S, V = tsvd(x, f.r)
    # TODO: the order of the following matrix products should depend on the shape of x
    M = S .* V'
    mul!(y, U, M)
    return R(0)
end

function prox_naive(f::IndBallRank, x, gamma)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        y = x
        return y, R(0)
    end
    F = svd(x)
    y = F.U[:,1:f.r]*(Diagonal(F.S[1:f.r])*F.V[:,1:f.r]')
    return y, R(0)
end
