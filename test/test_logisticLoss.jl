y = [1.0, -1.0, 1.0, -1.0, 1.0]
mu = 1.5

f = LogisticLoss(y, mu)

x = [-1.0, -2.0, 3.0, 2.0, 1.0]

f_x_1 = f(x)
grad_f_x, f_x_2 = gradient(f, x)

f_x_ref = 5.893450123044199
grad_f_x_ref = [-1.0965878679450072, 0.17880438303317633, -0.07113880976635019, 1.3211956169668235, -0.4034121320549927]

@test abs(f_x_1 - f_x_ref)/abs(f_x_ref) <= 1e-10
@test abs(f_x_2 - f_x_ref)/abs(f_x_ref) <= 1e-10
@test norm(grad_f_x - grad_f_x_ref, Inf)/norm(grad_f_x_ref, Inf) <= 1e-10

z1, f_z1 = prox(f, x)
grad_f_z1, = gradient(f, z1)

@test norm((x - z1)./1.0 - grad_f_z1, Inf)/norm(grad_f_z1, Inf) <= 1e-4

z2, f_z2 = prox(f, x, 2.0)
grad_f_z2, = gradient(f, z2)

@test norm((x - z2)./2.0 - grad_f_z2, Inf)/norm(grad_f_z2, Inf) <= 1e-4
