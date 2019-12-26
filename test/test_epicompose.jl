using Test
using LinearAlgebra
using SparseArrays
using ProximalOperators

@testset "Epicompose (Gram-diagonal)" begin

n = 5

for R in [Float64] # TODO: enable Float32?
    for T in [R, Complex{R}]
        A = randn(T, n, n)
        Q, _ = qr(A)
        mu = R(2)
        L = mu*Q
        Lfs = [
            (L, NormL1(R(1))),
            (sparse(L), NormL1(R(1))),
        ]
        for (L, f) in Lfs
            g = Epicompose(L, f, mu)
            x = randn(T, n)
            prox_test(g, x, R(2))
        end
    end
end

end

@testset "Epicompose (Quadratic)" begin

n = 5
m = 3

for R in [Float64] # TODO: enable Float32?
    W = randn(R, n, n)
    Q = W' * W
    q = randn(R, n)
    L = randn(R, m, n)
    Lfs = [
        (L, Quadratic(Q, q)),
        (sparse(L), Quadratic(sparse(Q), q)),
    ]
    for (L, f) in Lfs
        g = Epicompose(L, f)
        x = randn(R, m)
        prox_test(g, x, R(2))
    end
end

end
