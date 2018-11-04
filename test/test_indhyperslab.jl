using Test
using Random
using LinearAlgebra
using ProximalOperators

Random.seed!(0)

@testset "IndHyperslab" begin

for R in [Float16, Float32, Float64]
    for shape in [(5,), (3, 5), (3, 4, 5)]
        c = randn(R, shape)
        x = randn(R, shape)
        cx = dot(c, x)

        for (low, upp) in [(cx-R(1), cx+R(1)), (cx-R(2), cx-R(1)), (cx+R(1), cx+R(2))]
            f = IndHyperslab(low, c, upp)
            predicates_test(f)
            call_test(f, x)
            prox_test(f, x, R(0.5)+rand(R))
        end
    end
end

end
