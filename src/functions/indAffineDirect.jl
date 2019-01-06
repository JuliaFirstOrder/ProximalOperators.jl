### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a QR factorization of A'.

using LinearAlgebra: QRCompactWY

struct IndAffineDirect{R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization} <: IndAffine
    A::M
    b::V
    fact::F
    res::V
    function IndAffineDirect{R, T, M, V, F}(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization}
        if size(A, 1) > size(A, 2)
            error("A must be full row rank")
        end
        normrowsinv = 1 ./ vec(sqrt.(sum(abs2.(A); dims=2)))
        A = normrowsinv.*A # normalize rows of A
        b = normrowsinv.*b # and b accordingly
        fact = qr(M(A'))
        new(A, b, fact, similar(b))
    end
end

is_cone(f::IndAffineDirect) = norm(f.b) == 0.0

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: DenseMatrix{T}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, QRCompactWY{T, M}}(A, b)

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, I <: Integer, M <: SparseMatrixCSC{T, I}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, SuiteSparse.SPQR.Factorization{T}}(A, b)

IndAffineDirect(a::V, b::T) where {R <: Real, T <: RealOrComplex{R}, V <: AbstractVector{T}} = IndAffineDirect(reshape(a,1,:), [b])

function (f::IndAffineDirect{R, T, M, V, F})(x::V) where {R, T, M, V, F}
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    # the tolerance in the following line should be customizable
    if norm(f.res, Inf) <= 1e-12
        return R(0)
    end
    return typemax(R)
end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=R(1)) where {R, T, M, V, F <: QRCompactWY}
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    Rfact = view(f.fact.factors, 1:length(f.b), 1:length(f.b))
    LinearAlgebra.LAPACK.trtrs!('U', 'C', 'N', Rfact, f.res)
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', Rfact, f.res)
    mul!(y, adjoint(f.A), f.res)
    y .+= x
    return R(0)
end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=R(1)) where {R, T, M, V, F <: SuiteSparse.SPQR.Factorization}
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    ##############################################################################
    # We need to solve for z: AA'z = res
    # We have: QR=PA'S so A'=P'QRS' and AA'=SR'Q'PP'QRS'=SR'RS'
    # So: z = S'\R\R'\S\res
    # TODO: the following lines should be made more efficient
    temp = f.res[f.fact.pcol]
    temp = LowerTriangular(adjoint(f.fact.R))\temp
    temp = UpperTriangular(f.fact.R)\temp
    RRres = similar(temp)
    RRres[f.fact.pcol] .= temp
    ##############################################################################
    mul!(y, adjoint(f.A), RRres)
    y .+= x
    return R(0)
end

function prox_naive(f::IndAffineDirect, x::AbstractArray{T,1}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    y = x + f.A'*((f.A*f.A')\(f.b - f.A*x))
    return y, R(0)
end
