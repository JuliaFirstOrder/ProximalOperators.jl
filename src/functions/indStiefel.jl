# indicator of the Stiefel manifold

export IndStiefel

@doc raw"""
    IndStiefel()

Return the indicator of the Stiefel manifold
```math
S_{n,p} = \left\{ X \in \mathbb{F}^{n \times p} : X^*X = I \right\}.
```
where ``\mathbb{F}`` is the real or complex field, and parameters ``n`` and ``p``
are inferred from the matrix provided as input.
"""
struct IndStiefel end

is_set_indicator(f::Type{<:IndStiefel}) = true

function (::IndStiefel)(X)
    R = real(eltype(X))
    F = svd(X)
    if all(F.S .≈ R(1))
        return R(0)
    end
    return R(Inf)
end

function prox!(Y, ::IndStiefel, X, gamma)
    R = real(eltype(X))
    n, p = size(X)
    F = svd(X)
    U_sliced = view(F.U, :, 1:p)
    mul!(Y, U_sliced, F.Vt)
    return R(0)
end

function prox_naive(::IndStiefel, X, gamma)
    R = real(eltype(X))
    n, p = size(X)
    F = svd(X)
    Y = F.U[:, 1:p] * F.Vt
    return Y, R(0)
end
