
using Random
using ProximalOperators
using Test


@testset "NormL1plusL2 standard case" begin
    f = NormL1(1.0)
    g = NormL2(2.0)
    fplusg = NormL1plusL2(1.0, 2.0)
    
    x = randn(50)
    
    y1, f1 = prox(f, x)
    y2, f2 = prox(g, y1)
    
    y3, f3 = prox(fplusg, x)
    
    @test f2 ≈ f2
    @test y2 ≈ y3
end



@testset "NormL1plusL2 norm constructor" begin
    f = NormL1(1.0)
    g = NormL2(2.0)
    fplusg = NormL1plusL2(f, g)
    
    x = randn(50)
    
    y1, f1 = prox(f, x)
    y2, f2 = prox(g, y1)
    
    y3, f3 = prox(fplusg, x)
    
    @test f2 ≈ f2
    @test y2 ≈ y3
end

@testset "NormL1plusL2 vector case" begin
    λ1 = abs.(randn(50))

    f = NormL1(λ1)
    g = NormL2(2.0)
    
    fplusg = NormL1plusL2(λ1, 2.0)
    
    x = randn(50)
    
    y1, f1 = prox(f, x)
    y2, f2 = prox(g, y1)
    
    y3, f3 = prox(fplusg, x)
    
    @test f2 ≈ f2
    @test y2 ≈ y3
end