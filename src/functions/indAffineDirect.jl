### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a QR factorization of A'.

using LinearAlgebra: QRCompactWY

struct IndAffineDirect{M, V, F} <: IndAffine
    A::M
    b::V
    fact::F
    res::V
    function IndAffineDirect{M, V, F}(A::M, b::V) where {M, V, F}
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

IndAffineDirect(A::M, b::V) where {T, M <: DenseMatrix{T}, V} = IndAffineDirect{M, V, QRCompactWY{T, M}}(A, b)

IndAffineDirect(A::M, b::V) where {T, M <: SparseMatrixCSC{T}, V} = IndAffineDirect{M, V, SuiteSparse.SPQR.Factorization{T}}(A, b)

IndAffineDirect(a::V, b::T) where {T, V <: AbstractVector{T}} = IndAffineDirect(reshape(a,1,:), [b])

factorization_type(::IndAffineDirect{M, V, F}) where {M, V, F} = F

function (f::IndAffineDirect)(x)
    R = real(eltype(x))
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    # the tolerance in the following line should be customizable
    if norm(f.res, Inf) <= sqrt(eps(R))
        return R(0)
    end
    return typemax(R)
end

prox!(y, f::IndAffineDirect, x, gamma) = prox!(factorization_type(f), y, f, x, gamma)

function prox!(::Type{<:QRCompactWY}, y, f::IndAffineDirect, x, gamma)
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    Rfact = view(f.fact.factors, 1:length(f.b), 1:length(f.b))
    LinearAlgebra.LAPACK.trtrs!('U', 'C', 'N', Rfact, f.res)
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', Rfact, f.res)
    mul!(y, adjoint(f.A), f.res)
    y .+= x
    return real(eltype(x))(0)
end

function prox!(::Type{<:SuiteSparse.SPQR.Factorization}, y, f::IndAffineDirect, x, gamma)
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
    return real(eltype(x))(0)
end
