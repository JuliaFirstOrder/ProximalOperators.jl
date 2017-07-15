# Test with full matrices

n, k = 5, 4

A = randn(n, k)
Q = A*A'
q = randn(n)
f = Quadratic(Q, q)

predicates_test(f)

@test ProximalOperators.is_smooth(f) == true
@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_set(f) == false

x = randn(n)

grad_fx, fx = gradient(f, x)
@test abs(fx - 0.5*dot(x, Q*x) - dot(x, q)) <= 1e-12
@test norm(grad_fx - (Q*x + q), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

f = QuadraticIterative(Q, q)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

# Test with sparse matrices

Q = sparse(Q)
f = Quadratic(Q, q)

grad_fx, fx = gradient(f, x)
@test abs(fx - 0.5*dot(x, Q*x) - dot(x, q)) <= 1e-12
@test norm(grad_fx - (Q*x + q), Inf) <= 1e-12

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 0.8)

f = QuadraticIterative(Q, q)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.3)
