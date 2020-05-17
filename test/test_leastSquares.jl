using LinearAlgebra
using SparseArrays
using ProximalOperators
using Test

@testset "LeastSquares" begin

@testset "$(T), $(s), $(matrix_type), $(mode)" for (T, s, matrix_type, mode) in Iterators.product(
    [Float64, ComplexF64],
    [(10, 29), (29, 10)],
    [:dense, :sparse],
    [:direct, :iterative],
)

R = real(T)

A = if matrix_type == :sparse sparse(randn(T, s...)) else randn(T, s...) end
b = randn(T, s[1])
x = randn(T, s[2])

f = LeastSquares(A, b, iterative=(mode == :iterative))
predicates_test(f)

@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_set(f) == false

grad_fx, fx = gradient(f, x)
lsres = A*x - b
@test abs(fx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - (A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, R(1.5))

lam = R(0.1) + rand(R)
f = LeastSquares(A, b, lam, iterative=(mode == :iterative))
predicates_test(f)

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, R(2.1))

end

end