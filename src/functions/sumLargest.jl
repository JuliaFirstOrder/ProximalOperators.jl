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
SumLargest(k::I=1, lambda::R=1) where {I, R} = Postcompose(Conjugate(IndSimplex(k)), lambda)

function (f::Conjugate{<:IndSimplex})(x)
    if f.f.a == 1
        return maximum(x)
    end
    p = if ndims(x) == 1
        partialsortperm(x, 1:f.f.a, rev=true)
    else
        partialsortperm(x[:], 1:f.f.a, rev=true)
    end
    return sum(x[p])
end
