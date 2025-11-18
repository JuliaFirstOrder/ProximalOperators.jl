using ProximalOperators
using Test

@testset "PointwiseMinimum" begin

T = Float64

f = PointwiseMinimum(IndPoint(T[-1.0]), IndPoint(T[1.0]))
x = T[0.1]

predicates_test(f)
@test ProximalCore.is_set_indicator(f) == true
@test ProximalCore.is_cone_indicator(f) == false

y, fy = prox_test(f, x)
@test all(y .== T[1.0])
@test fy == T(0)

end
