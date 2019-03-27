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
    (solver) -> IndPolyhedral(l, A, solver=solver),
    (solver) -> IndPolyhedral(l, A, xmin, xmax, solver=solver),
    (solver) -> IndPolyhedral(A, u, solver=solver),
    (solver) -> IndPolyhedral(A, u, xmin, xmax, solver=solver),
    (solver) -> IndPolyhedral(l, A, u, solver=solver),
    (solver) -> IndPolyhedral(l, A, u, xmin, xmax, solver=solver),
]

constructors_negative = [
    (solver) -> IndPolyhedral(l, A, xmax, xmin, solver=solver),
    (solver) -> IndPolyhedral(A, u, xmax, xmin, solver=solver),
    (solver) -> IndPolyhedral(l, A, u, xmax, xmin, solver=solver),
]

# run positive tests

for constr in constructors_positive
    for solver in [:osqp, :qpdas]
        f = constr(solver)
        @test ProximalOperators.is_convex(f) == true
        @test ProximalOperators.is_set(f) == true
        fx = call_test(f, x)
        p, fp = prox_test(f, x)
    end
end

# run negative tests

for constr in constructors_negative
    for solver in [:osqp, :qpdas]
        @test_throws ErrorException constr(solver)
    end
end
