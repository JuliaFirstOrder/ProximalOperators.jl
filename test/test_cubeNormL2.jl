using Test
using Random
using ProximalOperators

Random.seed!(0)

@testset "CubeNormL2" begin

for R in [Float16, Float32, Float64]
    for T in [R, Complex{R}]
        for shape in [(5,), (3, 5), (3, 4, 5)]
            lambda = R(0.1) + 5*rand(R)
            f = CubeNormL2(lambda)
            predicates_test(f)
            x = randn(T, shape)
            call_test(f, x)
            prox_test(f, x, R(0.5)+rand(R))
        end
    end
end

end
