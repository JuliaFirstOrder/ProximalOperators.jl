using LinearAlgebra
using SparseArrays
using Test
using ProximalOperators

@testset "Quadratic" begin

# Test with full matrices

n, k = 5, 4

A = randn(n, k)
Q = A*A'
q = randn(n)
f = Quadratic(Q, q)
@test typeof(f) <: ProximalOperators.QuadraticDirect

predicates_test(f)

@test is_smooth(f) == true
@test is_quadratic(f) == true
@test is_set_indicator(f) == false

x = randn(n)

grad_fx, fx = gradient_test(f, x)
@test fx ≈ 0.5*dot(x, Q*x) + dot(x, q)
@test all(grad_fx .≈ (Q*x + q))

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

f = Quadratic(Q, q, iterative=true)
@test typeof(f) <: ProximalOperators.QuadraticIterative

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# Test with sparse matrices

Q = sparse(Q)
f = Quadratic(Q, q)
@test typeof(f) <: ProximalOperators.QuadraticDirect

grad_fx, fx = gradient_test(f, x)
@test fx ≈ 0.5*dot(x, Q*x) + dot(x, q)
@test all(grad_fx .≈ (Q*x + q))

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 0.8)

f = Quadratic(Q, q, iterative=true)
@test typeof(f) <: ProximalOperators.QuadraticIterative

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.3)

end