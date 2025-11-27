### CONCRETE TYPE: ITERATIVE PROX EVALUATION

struct IndAffineIterative{M, V} <: IndAffine
    A::M
    b::V
    res::V
    function IndAffineIterative{M, V}(A::M, b::V) where {M, V}
        if size(A,1) > size(A,2)
            error("A must be full row rank")
        end
        normrowsinv = 1 ./ vec(sqrt.(sum(abs2.(A); dims=2)))
        A = normrowsinv.*A # normalize rows of A
        b = normrowsinv.*b # and b accordingly
        new(A, b, similar(b))
    end
end

is_proximable(f::Type{<:IndAffineIterative}) = false

IndAffineIterative(A::M, b::V) where {M, V} = IndAffineIterative{M, V}(A, b)

function (f::IndAffineIterative{M, V})(x) where {M, V}
    R = real(eltype(x))
    mul!(f.res, f.A, x)
    f.res .= f.b .- f.res
    # the tolerance in the following line should be customizable
    if norm(f.res, Inf) <= sqrt(eps(R))
        return R(0)
    end
    return typemax(R)
end

function prox!(y, f::IndAffineIterative{M, V}, x, gamma) where {M, V}
    # Von Neumann's alternating projections
    R = real(eltype(x))
    y .= x
    for k = 1:1000
        maxres = R(0)
        for i in eachindex(f.b)
            resi = (f.b[i] - dot(f.A[i,:], y))
            y .= y + resi*f.A[i,:] # no need to divide: rows of A are normalized
            absresi = resi > 0 ? resi : -resi
            maxres = absresi > maxres ? absresi : maxres
        end
        if maxres < sqrt(eps(R))
            break
        end
    end
    return R(0)
end
