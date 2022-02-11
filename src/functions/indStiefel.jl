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

is_set(f::Type{<:IndStiefel}) = true

function (f::IndStiefel)(X::AbstractMatrix{T}) where {R <: Real, T <: Union{R, Complex{R}}}
    F = svd(X)
    if all(F.S .≈ R(1))
        return R(0)
    end
    return R(Inf)
end

function prox!(Y::AbstractMatrix{T}, f::IndStiefel, X::AbstractMatrix{T}, gamma::R) where {R <: Real, T <: Union{R, Complex{R}}}
    n, p = size(X)
    F = svd(X)
    U_sliced = view(F.U, :, 1:p)
    mul!(Y, U_sliced, F.Vt)
    return R(0)
end

function prox_naive(f::IndStiefel, X::AbstractMatrix{T}, gamma::R) where {R, T <: Union{R, Complex{R}}}
    n, p = size(X)
    F = svd(X)
    Y = F.U[:, 1:p] * F.Vt
    return Y, R(0)
end
