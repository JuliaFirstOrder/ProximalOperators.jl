srand(0)

# set dimensions

m, n = 25, 10

# pick random (nonempty) polyhedron

xmin = -ones(n)
xmax = +ones(n)
x0 = min.(xmax, max.(xmin, 10.*rand(n) .- 5.0))
A = randn(m, n)
u = A*x0 .+ 0.1
l = A*x0 .- 0.1

# pick random point

x = 10.*randn(n)
p = similar(x)

# l

f = IndPolyhedral(l, A)
p, fp = prox_test(f, x)

# l, xmin, xmax

f = IndPolyhedral(l, A, xmin, xmax)
p, fp = prox_test(f, x)

# u

f = IndPolyhedral(A, u)
p, fp = prox_test(f, x)

# u, xmin, xmax

f = IndPolyhedral(A, u, xmin, xmax)
p, fp = prox_test(f, x)

# l, u

f = IndPolyhedral(l, A, u)
p, fp = prox_test(f, x)

# l, u, xmin, xmax

f = IndPolyhedral(l, A, u, xmin, xmax)
p, fp = prox_test(f, x)
