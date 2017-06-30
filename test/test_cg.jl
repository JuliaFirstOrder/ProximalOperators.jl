# Test conjugate gradient routines

@printf("\nTesting CG routines\n")

n, m = 500, 1000
B = randn(n, m)
A = B*B';
x_star = randn(n)
b = A*x_star
x = similar(x_star)

it = ProximalOperators.cg!(x, A, 0.0, b, 1e-10)

@test norm(A*x - b) <= 1e-8
@test norm(x - x_star) <= 1e-6

c = A*x_star + 0.5*x_star

it = ProximalOperators.cg!(x, A, 0.5, c, 1e-10)

@test norm(A*x + 0.5*x - c) <= 1e-8
@test norm(x - x_star) <= 1e-6
