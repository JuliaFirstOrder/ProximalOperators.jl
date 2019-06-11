using ProximalOperators
using Test

@testset "PointwiseMinimum" begin

T = Float64

f = PointwiseMinimum(IndPoint(T[-1.0]), IndPoint(T[1.0]))
x = T[0.1]

predicates_test(f)
@test ProximalOperators.is_set(f) == true
@test ProximalOperators.is_cone(f) == false

y, fy = prox_test(f, x)
@test all(y .== T[1.0])
@test fy == T(0)

end
