# Postcompose HuberLoss

f = HuberLoss(1.0, 1.0)
g = Postcompose(f, 2.5)
h = HuberLoss(1.0, 2.5)

x = randn(10)
x = 0.5*x/norm(x)

gx = call_test(g, x)
hx = call_test(h, x)

@test abs(gx-hx)/(1+abs(gx)) <= 1e-12

grad_gx, gx1 = gradient(g, x)
grad_hx, hx1 = gradient(h, x)

@test abs(gx1-gx)/(1+abs(gx)) <= 1e-12
@test abs(hx1-hx)/(1+abs(hx)) <= 1e-12
@test norm(grad_gx-grad_hx, Inf)/(1+norm(grad_gx, Inf)) <= 1e-12

yg, gyg = prox_test(g, x, 1.3)
yh, hyh = prox_test(h, x, 1.3)

@test abs(gyg-hyh)/(1+abs(gyg)) <= 1e-12
@test norm(yg-yh, Inf)/(1+norm(yg, Inf)) <= 1e-12

x = randn(10)
x = 1.2*x/norm(x)

gx = call_test(g, x)
hx = call_test(h, x)

@test abs(gx-hx)/(1+abs(gx)) <= 1e-12

grad_gx, gx1 = gradient(g, x)
grad_hx, hx1 = gradient(h, x)

@test abs(gx1-gx)/(1+abs(gx)) <= 1e-12
@test abs(hx1-hx)/(1+abs(hx)) <= 1e-12
@test norm(grad_gx-grad_hx, Inf)/(1+norm(grad_gx, Inf)) <= 1e-12

yg, gyg = prox_test(g, x, 1.3)
yh, hyh = prox_test(h, x, 1.3)

@test abs(gyg-hyh)/(1+abs(gyg)) <= 1e-12
@test norm(yg-yh, Inf)/(1+norm(yg, Inf)) <= 1e-12
