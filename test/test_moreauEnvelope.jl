using Test
using LinearAlgebra

@testset "Moreau envelope" begin

@testset "Box indicator" for R in [Float32, Float64]

    f = IndBox(-1, 1)

    for g in [
        MoreauEnvelope(f),
        MoreauEnvelope(f, R(0.01))
    ]

        predicates_test(g)

        @test is_smooth(g) == true
        @test is_quadratic(g) == false
        @test is_set_indicator(g) == false

        x = R[1.0, 2.0, 3.0, 4.0, 5.0]

        grad_g_x, g_x = gradient_test(g, x)

        y, g_y = prox_test(g, x, R(1/2))
        grad_g_y, _ = gradient_test(g, y)

        @test y + grad_g_y / 2 ≈ x
        @test g(y) ≈ g_y
    end

end

@testset "L2 norm" for R in [Float32, Float64]

    for (g, h) in [
        (MoreauEnvelope(NormL2()), HuberLoss()),
        (MoreauEnvelope(NormL2(R(1)), R(0.1)), HuberLoss(R(0.1), R(1)/R(0.1)))
    ]

        predicates_test(g)

        @test is_smooth(g) == true
        @test is_quadratic(g) == false
        @test is_set_indicator(g) == false

        x = R[1.0, 2.0, 3.0, 4.0, 5.0]

        @test g(x) ≈ h(x)

        grad_g_x, g_x = gradient_test(g, x)
        grad_h_x, h_x = gradient_test(h, x)

        @test g_x ≈ g(x)
        @test h_x ≈ h(x)
        @test all(grad_g_x .≈ grad_h_x)

        y, g_y = prox_test(g, x, R(1/2))
        grad_g_y, _ = gradient_test(g, y)

        @test y + grad_g_y / 2 ≈ x
        @test g(y) ≈ g_y
    
    end

end

end