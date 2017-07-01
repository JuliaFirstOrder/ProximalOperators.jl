# Wide full matrix

m, n = 10, 30
A = randn(m, n)
b = randn(m)
f = LeastSquares(A, b)
x = randn(n)

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

grad_fx, fx = gradient(f, x)
@test abs(fx - (lam/2)*norm(lsres)^2) <= 1e-12
@test norm(grad_fx - lam*(A'*lsres), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# Wide sparse matrix

m, n = 10, 30
A = sprandn(m, n, 0.5)
b = randn(m)
f = LeastSquares(A, b)
x = randn(n)

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

# Tall full matrix

m, n = 30, 10
A = randn(m, n)
b = randn(m)
f = LeastSquares(A, b)
@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true
x = randn(n)

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

# Wide sparse matrix

m, n = 30, 10
A = sprandn(m, n, 0.5)
b = randn(m)
f = LeastSquares(A, b)
@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_generalized_quadratic(f) == true
@test ProximalOperators.is_convex(f) == true
x = randn(n)

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
