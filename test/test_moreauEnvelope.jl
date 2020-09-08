using Test
using LinearAlgebra

@testset "Moreau envelope" begin

@testset "Box indicator" begin

    f = IndBox(-1, 1)
    g = MoreauEnvelope(f, 1e-2)

    predicates_test(g)

    @test ProximalOperators.is_smooth(g) == true
    @test ProximalOperators.is_quadratic(g) == false
    @test ProximalOperators.is_set(g) == false

    x = [1.0, 2.0, 3.0, 4.0, 5.0]

    grad_g_x, g_x = gradient(g, x)

    y, g_y = prox(g, x, 0.5)
    grad_g_y, _ = gradient(g, y)

    @test y + 0.5 * grad_g_y ≈ x
    @test g(y) ≈ g_y

end

@testset "L2 norm" begin

    rho = 1e-1
    mu = 1e0
    f = NormL2(mu)
    g = MoreauEnvelope(f, rho)
    h = HuberLoss(rho, mu/rho)

    predicates_test(g)

    @test ProximalOperators.is_smooth(g) == true
    @test ProximalOperators.is_quadratic(g) == false
    @test ProximalOperators.is_set(g) == false

    x = [1.0, 2.0, 3.0, 4.0, 5.0]

    @test abs(g(x) - h(x)) <= 1e-12

    grad_g_x, g_x = gradient(g, x)
    grad_h_x, h_x = gradient(h, x)

    @test abs(g_x - g(x)) <= 1e-12
    @test abs(h_x - h(x)) <= 1e-12
    @test norm(grad_g_x - grad_h_x, Inf) <= 1e-12

    y, g_y = prox(g, x, 0.5)
    grad_g_y, _ = gradient(g, y)

    @test y + 0.5 * grad_g_y ≈ x
    @test g(y) ≈ g_y

end

end