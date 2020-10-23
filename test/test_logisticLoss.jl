using LinearAlgebra
using Test
using ProximalOperators

@testset "LogisticLoss" for T in [Float32, Float64]

y = T[1.0, -1.0, 1.0, -1.0, 1.0]
mu = T(1.5)

f = LogisticLoss(y, mu)

x = T[-1.0, -2.0, 3.0, 2.0, 1.0]

f_x_1 = f(x)

@test typeof(f_x_1) == T

grad_f_x, f_x_2 = gradient(f, x)

f_x_ref = 5.893450123044199
grad_f_x_ref = [-1.0965878679450072, 0.17880438303317633, -0.07113880976635019, 1.3211956169668235, -0.4034121320549927]

@test f_x_1 ≈ f_x_ref
@test f_x_2 ≈ f_x_ref
@test all(grad_f_x .≈ grad_f_x_ref)

z1, f_z1 = prox(f, x)
grad_f_z1, = gradient(f, z1)

@test typeof(f_z1) == T
@test norm((x - z1)./1.0 - grad_f_z1, Inf)/norm(grad_f_z1, Inf) <= 1e-4

z2, f_z2 = prox(f, x, T(2.0))
grad_f_z2, = gradient(f, z2)

@test typeof(f_z2) == T
@test norm((x - z2)./2.0 - grad_f_z2, Inf)/norm(grad_f_z2, Inf) <= 1e-4

end