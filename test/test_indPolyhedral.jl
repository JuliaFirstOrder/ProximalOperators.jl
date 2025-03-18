using ProximalOperators
using Test

@testset "IndPolyhedral" begin

# set dimensions

m, n = 25, 10

# pick random (nonempty) polyhedron

xmin = -ones(n)
xmax = +ones(n)
x0 = min.(xmax, max.(xmin, 10 .* rand(n) .- 5.0))
A = randn(m, n)
u = A*x0 .+ 0.1
l = A*x0 .- 0.1

# pick random point

x = 10 .* randn(n)
p = similar(x)

@testset "valid" for constr in [
    () -> IndPolyhedral(l, A),
    () -> IndPolyhedral(l, A, xmin, xmax),
    () -> IndPolyhedral(A, u),
    () -> IndPolyhedral(A, u, xmin, xmax),
    () -> IndPolyhedral(l, A, u),
    () -> IndPolyhedral(l, A, u, xmin, xmax),
]
    f = constr()
    @test ProximalCore.is_convex(f) == true
    @test ProximalCore.is_set_indicator(f) == true
    fx = call_test(f, x)
    p, fp = prox_test(f, x)
end

@testset "invalid" for constr in [
    () -> IndPolyhedral(l, A, xmax, xmin),
    () -> IndPolyhedral(A, u, xmax, xmin),
    () -> IndPolyhedral(l, A, u, xmax, xmin),
]
    @test_throws ErrorException constr()
end

end
