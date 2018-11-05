using Test
using Random

Random.seed!(0)

@testset "Linear" begin

for R in [Float16, Float32, Float64]
    for shape in [(5,), (3, 5), (3, 4, 5)]
        c = randn(R, shape)
        f = Linear(c)
        predicates_test(f)
        x = randn(R, shape)
        @test gradient(f, x) == (c, f(x))
        call_test(f, x)
        prox_test(f, x, R(0.5)+rand(R))
    end
end

end
