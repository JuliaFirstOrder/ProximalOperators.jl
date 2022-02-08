# Sum of the largest k components

# export SumLargest

# TODO: SumLargest(r) is (the postcomposition of) the conjugate of
#  (1) ind{0 <= x <= 1 : sum(x) = r},
# where instead IndSimplex(r) corresponds to
#  (2) ind{0 <= x : sum(x) = r}.
# Therefore SumLargest(r) now is only correct when r = 1.
# To make SumLargest correct we should extend IndSimplex to allow for that
# additional bound in its definition, by adding a second argument to the
# constructor. Then in the following line we should replace IndSimplex(k)
# with IndSimplex(k, 1.0). Note that (1) is proper only if x ∈ Rⁿ for n ⩾ r.

"""
    SumLargest(k::Integer=1, λ::Real=1)

Return the function `g(x) = λ⋅sum(x_[1], ..., x_[k])`, for an integer k ⩾ 1 and `λ ⩾ 0`.
"""
SumLargest(k::I=1, lambda::R=1) where {I <: Integer, R <: Real} = Postcompose(Conjugate(IndSimplex(k)), lambda)

function (f::Conjugate{IndSimplex{I}})(x::AbstractArray{S}) where {I <: Integer, S <: Real}
    if f.f.a == 1
        return maximum(x)
    end
    v = zero(S)
    if ndims(x) == 1
        p = partialsortperm(x, 1:f.f.a, rev=true)
        v = sum(x[p])
    else
        p = partialsortperm(x[:], 1:f.f.a, rev=true)
        v = sum(x[p])
    end
    return v
end
