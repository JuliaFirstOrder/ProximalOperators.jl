using Test
using Random

Random.seed!(0)

for R in [Float16, Float32, Float64]
    for T in [R, Complex{R}]
        for shape in [(5,), (3, 5), (3, 4, 5)]
            # Real
            c = randn(T, shape)
            f = Linear(c)
            predicates_test(f)
            x = randn(T, shape)
            @test gradient(f, x) == (c, f(x))
            call_test(f, x)
            prox_test(f, x, R(0.5)+rand(R))
        end
    end
end
