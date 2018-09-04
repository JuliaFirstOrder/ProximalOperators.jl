using Random

Random.seed!(0)

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

# define test cases

constructors_positive = [
    () -> IndPolyhedral(l, A),
    () -> IndPolyhedral(l, A, xmin, xmax),
    () -> IndPolyhedral(A, u),
    () -> IndPolyhedral(A, u, xmin, xmax),
    () -> IndPolyhedral(l, A, u),
    () -> IndPolyhedral(l, A, u, xmin, xmax),
]

constructors_negative = [
    () -> IndPolyhedral(l, A, xmax, xmin),
    () -> IndPolyhedral(A, u, xmax, xmin),
    () -> IndPolyhedral(l, A, u, xmax, xmin),
]

# run positive tests

for constr in constructors_positive
    f = constr()
    @test ProximalOperators.is_convex(f) == true
    @test ProximalOperators.is_set(f) == true
    fx = call_test(f, x)
    p, fp = prox_test(f, x)
end

# run negative tests

for constr in constructors_negative
    @test_throws ErrorException constr()
end
