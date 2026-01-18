using LinearAlgebra
using ProximalOperators
using Test

@testset "HuberLoss" begin

f = HuberLoss(1.5, 0.7)

predicates_test(f)

@test is_smooth(f) == true
@test is_quadratic(f) == false
@test is_set_indicator(f) == false

x = randn(10)
x = 1.6*x/norm(x)

call_test(f, x)
prox_test(f, x, 1.3)
grad_fx, fx = gradient_test(f, x)

@test abs(fx - f(x)) <= 1e-12
@test norm(0.7*1.5*x/norm(x) - grad_fx, Inf) <= 1e-12

x = randn(10)
x = 1.4*x/norm(x)

call_test(f, x)
prox_test(f, x, 0.9)
grad_fx, fx = gradient_test(f, x)

@test abs(fx - f(x)) <= 1e-12
@test norm(0.7*x - grad_fx, Inf) <= 1e-12

end