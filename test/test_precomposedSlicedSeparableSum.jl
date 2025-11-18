# x = (randn(10), randn(10))
# norm(x[1], 1) + norm(A2[1:5, 1:5] * x[2][1:5], 2) + norm(A2[6:10, 6:10] * x[2][6:10], 2)^2

@testset "PrecomposedSlicedSeparableSum" begin

fs = (NormL1(), NormL2(), SqrNormL2())

A1 = (Diagonal(ones(10)), nothing)
F = qr(randn(5, 5))
A2 = (nothing, Matrix(F.Q))
F = qr(randn(5, 5))
A3 = (nothing, Matrix(F.Q))
mu = rand(5)
A3[2] .*= reshape(mu, 5, 1)
ops = (A1, A2, A3)

idxs = ((Colon(), nothing), (nothing, 1:5), (nothing, 6:10))
μs = (1.0, 1.0, mu)

AAc2 = A2[2] * A2[2]'
@test AAc2 ≈ I
AAc3 = A3[2] * A3[2]'
@test AAc3 ≈ Diagonal(mu) .^ 2

import Base: *, adjoint
# mock HCAT from AbstractOperators

function Base.:*(A::Tuple, x::Tuple)
    codomain_size = size(A[findfirst(a -> !(a isa Nothing), A)], 1)
    out = zeros(codomain_size)
    for i in eachindex(A)
        if A[i] isa Nothing
            continue
        end
        out .+= A[i] * x[i]
    end
    return out
end
function Base.:*(A::Tuple, x::AbstractArray) # adjoint operation
    domain_size = size(A[findfirst(a -> !(a isa Nothing), A)], 2)
    return mul!(Tuple(zeros(domain_size) for _ in eachindex(A)), A, x)
end
function LinearAlgebra.mul!(out::Tuple, A::Tuple, x::AbstractArray)
    for i in eachindex(A)
        if A[i] isa Nothing
            out[i] .= 0
        else
            out[i] .+= A[i]' * x
        end
    end
    return out
end
Base.adjoint(A::Tuple) = A


f = PrecomposedSlicedSeparableSum(fs, idxs, ops, μs)
x = (randn(10), rand(10))
y = (zeros(10), zeros(10))
fy = prox!(y, f, x, 1.0)
yn, fyn = ProximalOperators.prox_naive(f, x, 1.0)
y1, fy1 = prox(NormL1(), x[1], 1.0)
y2, fy2 = prox(Precompose(NormL2(), A2[2], 1), x[2][1:5], 1.0)
y3, fy3 = prox(Precompose(SqrNormL2(), A3[2], mu), x[2][6:10], 1.0)

@show f(y), fy
@show norm(y)
@test abs(fyn-fy)<1e-11
@test norm(yn[1]-y[1])+norm(yn[2]-y[2])<1e-11
@test abs((fy1+fy2+fy3)-fy)<1e-11
@test norm(y[1] - y1) < 1e-11
@test norm(y[2][1:5] - y2) < 1e-11
@test norm(y[2][6:10] - y3) < 1e-11

end
