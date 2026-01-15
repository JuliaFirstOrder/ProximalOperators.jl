using LinearAlgebra
using SparseArrays
using ProximalOperators
using Test

@testset "LeastSquares" begin

@testset "$(T), $(s), $(matrix_type), $(mode)" for (T, s, matrix_type, mode) in Iterators.product(
    [Float64, ComplexF64],
    [(10, 29), (29, 10), (10, 29, 3), (29, 10, 3)],
    [:dense, :sparse],
    [:direct, :iterative],
)

if mode == :iterative && length(s) == 3
    # FIXME this case is currently not supported due to cg! in IterativeSolvers
    # See https://github.com/JuliaMath/IterativeSolvers.jl/issues/248
    # The fix is simple, but affects a lot of solvers in IterativeSolvers 
    # Maybe we can use our own CG here and drop the dependency
    continue
end

R = real(T)

shape_A = s[1:2]
shape_b = if length(s) == 2 s[1] else s[[1, 3]] end
shape_x = if length(s) == 2 s[2] else s[2:3] end

A = if matrix_type == :sparse sparse(randn(T, shape_A...)) else randn(T, shape_A...) end
b = randn(T, shape_b...)
x = randn(T, shape_x...)

f = LeastSquares(A, b, iterative=(mode == :iterative))
predicates_test(f)

@test is_smooth(f) == true
@test is_quadratic(f) == true
@test is_generalized_quadratic(f) == true
@test is_set_indicator(f) == false

grad_fx, fx = gradient_test(f, x)
lsres = A*x - b
@test fx ≈ 0.5*norm(lsres)^2
@test all(grad_fx .≈ (A'*lsres))

call_test(f, x)
prox_test(f, x)
prox_test(f, x, R(1.5))

lam = R(0.1) + rand(R)
f = LeastSquares(A, b, lam, iterative=(mode == :iterative))
predicates_test(f)

grad_fx, fx = gradient_test(f, x)
@test fx ≈ (lam/2)*norm(lsres)^2
@test all(grad_fx .≈ lam*(A'*lsres))

call_test(f, x)
prox_test(f, x)
prox_test(f, x, R(2.1))

end

end