using LinearAlgebra
using SparseArrays
using Random
using ProximalOperators
using Test

@testset "IndAffine" begin

# Full matrix

m, n = 10, 30
A = randn(m, n)
b = randn(m)
f = IndAffine(A, b)
x = randn(n)

predicates_test(f)

@test ProximalCore.is_smooth(f) == false
@test ProximalCore.is_quadratic(f) == false
@test ProximalCore.is_generalized_quadratic(f) == true
@test ProximalCore.is_set_indicator(f) == true

call_test(f, x)
y, fy = prox_test(f, x)

@test f(y) == 0.0

# Sparse matrix

m, n = 10, 30
A = sprandn(m, n, 0.5)
b = randn(m)
f = IndAffine(A, b)
x = randn(n)

call_test(f, x)
y, fy = prox_test(f, x)

@test f(y) == 0.0

# Iterative version

m, n = 200, 500
A = sprandn(m, n, 0.5)
b = randn(m)
f = IndAffine(A, b; iterative=true)
x = randn(n)

call_test(f, x)
y, fy = prox_test(f, x)

@test f(y) == 0.0

end