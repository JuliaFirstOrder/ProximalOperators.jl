n, k = 5, 4

A = randn(n, k)
Q = A*A'
q = randn(n)
f = Quadratic(Q, q)
x = randn(n)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.5)

f = Quadratic(Q, q, true)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 2.1)

Q = sparse(Q)
f = Quadratic(Q, q)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 0.8)

f = Quadratic(Q, q, true)

call_test(f, x)
prox_test(f, x)
prox_test(f, x, 1.3)
