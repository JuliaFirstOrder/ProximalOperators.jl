using LinearAlgebra
using SparseArrays
using Random

Random.seed!(0)

# Wide full matrix

m, n = 10, 30
A = randn(m, n)
b = randn(m)
x = randn(n)

f = LeastSquares(A, b)
predicates_test(f)

@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_set(f) == false

grad_fx, fx = gradient(f, x)
lsres = A*x - b
@test abs(fx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - (A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

lam = 0.1 + rand()
f = LeastSquares(A, b, lam)
predicates_test(f)

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# ...iterative

g = LeastSquares(A, b; iterative=true)
predicates_test(g)

@test ProximalOperators.is_smooth(g) == true
@test ProximalOperators.is_quadratic(g) == true
@test ProximalOperators.is_set(g) == false

grad_gx, gx = gradient(g, x)
lsres = A*x - b
@test abs(gx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - (A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 1.5)

lam = 0.1 + rand()
g = LeastSquares(A, b, lam; iterative=true)
predicates_test(g)

grad_gx, gx = gradient(g, x)
@test abs(gx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - lam*(A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 2.1)

# Wide sparse matrix

m, n = 10, 30
A = sprandn(m, n, 0.5)
b = randn(m)
x = randn(n)

f = LeastSquares(A, b)

grad_fx, fx = gradient(f, x)
lsres = A*x - b
@test abs(fx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - (A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

lam = 0.1 + rand()
f = LeastSquares(A, b, lam)

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# ...iterative

g = LeastSquares(A, b; iterative=true)

grad_gx, gx = gradient(g, x)
lsres = A*x - b
@test abs(gx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - (A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 1.5)

lam = 0.1 + rand()
g = LeastSquares(A, b, lam; iterative=true)

grad_gx, gx = gradient(g, x)
@test abs(gx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - lam*(A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 2.1)

# Tall full matrix

m, n = 30, 10
A = randn(m, n)
b = randn(m)
x = randn(n)

f = LeastSquares(A, b)
@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true

grad_fx, fx = gradient(f, x)
lsres = A*x - b
@test abs(fx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - (A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

lam = 0.1 + rand()
f = LeastSquares(A, b, lam)
@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# ...iterative

g = LeastSquares(A, b; iterative=true)
@test ProximalOperators.is_smooth(g) == true
@test ProximalOperators.is_quadratic(g) == true
@test ProximalOperators.is_generalized_quadratic(g) == true
@test ProximalOperators.is_convex(g) == true

grad_gx, gx = gradient(g, x)
lsres = A*x - b
@test abs(gx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - (A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 1.5)

lam = 0.1 + rand()
g = LeastSquares(A, b, lam; iterative=true)
@test ProximalOperators.is_smooth(g) == true
@test ProximalOperators.is_quadratic(g) == true
@test ProximalOperators.is_generalized_quadratic(g) == true
@test ProximalOperators.is_convex(g) == true

grad_gx, gx = gradient(g, x)
@test abs(gx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - lam*(A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 2.1)

# Tall sparse matrix

m, n = 30, 10
A = sprandn(m, n, 0.5)
b = randn(m)
x = randn(n)

f = LeastSquares(A, b)

@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true

grad_fx, fx = gradient(f, x)
lsres = A*x - b
@test abs(fx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - (A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

lam = 0.1 + rand()
f = LeastSquares(A, b, lam)
@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# ...iterative

g = LeastSquares(A, b; iterative=true)

@test ProximalOperators.is_smooth(g) == true
@test ProximalOperators.is_quadratic(g) == true
@test ProximalOperators.is_generalized_quadratic(g) == true
@test ProximalOperators.is_convex(g) == true

grad_gx, gx = gradient(g, x)
lsres = A*x - b
@test abs(gx - 0.5*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - (A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 1.5)

lam = 0.1 + rand()
g = LeastSquares(A, b, lam; iterative=true)
@test ProximalOperators.is_smooth(g) == true
@test ProximalOperators.is_quadratic(g) == true
@test ProximalOperators.is_generalized_quadratic(g) == true
@test ProximalOperators.is_convex(g) == true

grad_gx, gx = gradient(g, x)
@test abs(gx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_gx - lam*(A'*lsres), Inf) <= 1e-12

call_test(g, x)
prox_test(g, x)
prox_test(g, x, 2.1)
