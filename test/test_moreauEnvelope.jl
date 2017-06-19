# Moreau envelope of box indicator

f = IndBox(-1, 1)
g = MoreauEnvelope(f, 1e-2)

x = [1.0, 2.0, 3.0, 4.0, 5.0]

grad_g_x, g_x = gradient(g, x)

# Moreau envelope of L2 norm = (circulant) Huber loss (for some parameters)

rho = 1e-1
mu = 1e0
f = NormL2(mu)
g = MoreauEnvelope(f, rho)
h = HuberLoss(rho, mu/rho)

x = [1.0, 2.0, 3.0, 4.0, 5.0]

@test abs(g(x) - h(x)) <= 1e-12

grad_g_x, g_x = gradient(g, x)
grad_h_x, h_x = gradient(h, x)

@test abs(g_x - g(x)) <= 1e-12
@test abs(h_x - h(x)) <= 1e-12
@test norm(grad_g_x - grad_h_x, Inf) <= 1e-12
