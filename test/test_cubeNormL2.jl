using Test
using Random
using ProximalOperators

@testset "CubeNormL2" begin

for R in [Float16, Float32, Float64]
    for T in [R, Complex{R}]
        for shape in [(5,), (3, 5), (3, 4, 5)]
            lambda = R(0.1) + 5*rand(R)
            f = CubeNormL2(lambda)
            predicates_test(f)
            x = randn(T, shape)
            call_test(f, x)
            gamma = R(0.5)+rand(R)
            y, f_y = prox_test(f, x, gamma)
            grad_f_y, f_y = gradient_test(f, y)
            @test grad_f_y ≈ (x - y)/gamma
        end
    end
end

end
