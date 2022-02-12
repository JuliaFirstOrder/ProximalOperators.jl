# nuclear Norm (times a constant)

export NuclearNorm

"""
    NuclearNorm(λ=1)

Return the nuclear norm
```math
f(X) = \\|X\\|_* = λ ∑_i σ_i(X),
```
where `λ` is a positive parameter and ``σ_i(X)`` is ``i``-th singular value of matrix ``X``.
"""
struct NuclearNorm{R}
    lambda::R
    function NuclearNorm{R}(lambda::R) where {R}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:NuclearNorm}) = true

NuclearNorm(lambda::R=1) where {R} = NuclearNorm{R}(lambda)

function (f::NuclearNorm)(X)
    F = svd(X)
    return f.lambda * sum(F.S)
end

function prox!(Y, f::NuclearNorm, X, gamma)
    R = real(eltype(X))
    F = svd(X)
    S_thresh = max.(R(0), F.S .- f.lambda*gamma)
    rankY = findfirst(S_thresh .== R(0))
    if rankY === nothing
        rankY = minimum(size(X))
    end
    Vt_thresh = view(F.Vt, 1:rankY, :)
    U_thresh = view(F.U, :, 1:rankY)
    # TODO: the order of the following matrix products should depend on the shape of x
    M = S_thresh[1:rankY] .* Vt_thresh
    mul!(Y, U_thresh, M)
    return f.lambda * sum(S_thresh)
end

function prox_naive(f::NuclearNorm, X, gamma)
    F = svd(X)
    S = max.(0, F.S .- f.lambda*gamma)
    Y = F.U * (Diagonal(S) * F.Vt)
    return Y, f.lambda * sum(S)
end
