srand(123)

x = randn(10)
X = randn(10,10)+im*randn(10,10)
y0 = randn(size(x))
Y0 = randn(size(X))+im*randn(size(X))
y, Y = copy(y0), copy(Y0)

lambdas = (abs.(randn(size(x))), 0.1)
prox_col = (NormL1(lambdas[1]), NormL2(lambdas[2]))

# testing constructors

f = SeparableSum(prox_col)

println(f)

y, fy = prox(f, (x, X), 1.)

y1, fy1 = prox(prox_col[1], x, 1.)
y2, fy2 = prox(prox_col[2], X, 1.)

@test norm((fy1+fy2)-fy) <= 1e-11
@test norm(y[1]-y1) <= 1e-11
@test norm(y[2]-y2) <= 1e-11

y, fy = prox(f, (x, X), (0.5, 1.3))

y1, fy1 = prox(prox_col[1], x, 0.5)
y2, fy2 = prox(prox_col[2], X, 1.3)

@test norm((fy1+fy2)-fy) <= 1e-11
@test norm(y[1]-y1) <= 1e-11
@test norm(y[2]-y2) <= 1e-11

gamma1 = 0.5+rand(10)

y, fy = prox(f, (x, X), (gamma1, 1.3))

y1, fy1 = prox(prox_col[1], x, gamma1)
y2, fy2 = prox(prox_col[2], X, 1.3)

@test norm((fy1+fy2)-fy) <= 1e-11
@test norm(y[1]-y1) <= 1e-11
@test norm(y[2]-y2) <= 1e-11
