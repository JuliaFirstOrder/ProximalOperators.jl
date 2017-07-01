# Nonsmooth + regularization

mu, lam = 2.0, 3.0

f = NormL1(mu)
g = Regularize(f, lam)
h = ElasticNet(mu, lam)

x = randn(10)

gx = call_test(g, x)
hx = call_test(h, x)

@test abs(gx - hx)/(1+abs(gx)) <= 1e-12

yg, gy = prox_test(g, x, 0.5)
yh, hy = prox_test(h, x, 0.5)

@test abs(gy - hy)/(1+abs(gy)) <= 1e-12
@test norm(yg - yh, Inf)/(1+norm(yg, Inf)) <= 1e-12

# Smooth + regularization (test also gradient)

m, n = 10, 20
lam = 1.5

A = randn(m, n)
b = randn(m)

f = LeastSquares(A, b)
g = Regularize(f, lam)

x = randn(n)
res = A*x-b

gx = call_test(g, x)

@test abs(0.5*norm(res)^2 + (0.5*lam)*norm(x)^2 - gx)/(1+abs(gx)) <= 1e-12

prox_test(g, x, 0.7)
grad_gx, gx1 = gradient(g, x)

@test abs(gx - gx1)/(1+abs(gx)) <= 1e-12
@test norm(grad_gx - A'*(A*x - b) - lam*x, Inf)/(1+norm(grad_gx, Inf)) <= 1e-12
