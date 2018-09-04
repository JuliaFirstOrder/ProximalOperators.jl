using LinearAlgebra

# Indicator of L1 norm ball composed with orthogonal matrix

f = IndBallL1()
A = randn(10, 10)
F = qr(A)
Q = Matrix(F.Q)

@test Q'*Q ≈ I
@test Q*Q' ≈ I

g = Precompose(f, Q, 1.0)

predicates_test(g)

@test ProximalOperators.is_smooth(g) == false
@test ProximalOperators.is_quadratic(g) == false
@test ProximalOperators.is_set(g) == true

x = randn(10)

call_test(g, x)
prox_test(g, x, 1.0)

# Larger example

A = randn(500, 500)
F = qr(A)
Q = Matrix(F.Q)

@test Q'*Q ≈ I
@test Q*Q' ≈ I

g = Precompose(f, Q, 1.0)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L1 norm composed with multiple of orthogonal matrix

f = NormL1()
A = randn(50, 50)
F = qr(A)
Q = Matrix(F.Q)

@test norm(Q'*Q - I) <= 1e-12
@test norm(Q*Q' - I) <= 1e-12

g = Precompose(f, 3.0*Q, 9.0)

x = randn(50)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with multiple of orthogonal matrix

f = NormL2()
A = randn(500, 500)
F = qr(A)
Q = Matrix(F.Q)

@test norm(Q'*Q - I) <= 1e-12
@test norm(Q*Q' - I) <= 1e-12

g = Precompose(f, 0.9*Q, 0.9^2)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with orthogonal matrix + translation

f = NormL2()
A = randn(500, 500)
b = randn(500)
F = qr(A)
Q = Matrix(F.Q)

@test norm(Q'*Q - I) <= 1e-12
@test norm(Q*Q' - I) <= 1e-12

g = Precompose(f, Q, 1.0, -b)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with diagonal matrix + translation
# checking that Precompose and PrecomposeDiagonal agree

f = NormL2()
A = Diagonal(3.0*ones(500))
b = randn(500)

g1 = Precompose(f, A, 9.0, -b)
g2 = PrecomposeDiagonal(f, 3.0, -b)

x = randn(500)

call_test(g1, x)
y1, gy1 = prox_test(g1, x, 1.0)

call_test(g2, x)
y2, gy2 = prox_test(g2, x, 1.0)

@test abs(gy1 - gy2) <= (1 + abs(gy1))*1e-12
@test norm(y1 - y2) <= (1 + norm(y1))*1e-12

# Squared L2 norm composed with diagonal matrix + translation
# checking that Precompose and PrecomposeDiagonal agree
# checking that weighted squared L2 norm + Translate agrees too

f = SqrNormL2()
diagA = [rand(250); -rand(250)]
A = Diagonal(diagA)
b = randn(500)

g1 = Precompose(f, A, diagA .* diagA, -diagA .* b)
g2 = PrecomposeDiagonal(f, diagA, -diagA .* b)
g3 = Translate(SqrNormL2(diagA .* diagA), -b)

x = randn(500)

gx = 0.5*sum((diagA .* diagA) .* (x-b).^2)
grad_gx = diagA.*diagA.*(x - b)

@test abs(g1(x) - gx)/(1+abs(gx)) <= 1e-14
@test abs(g2(x) - gx)/(1+abs(gx)) <= 1e-14
@test abs(g3(x) - gx)/(1+abs(gx)) <= 1e-14

call_test(g1, x)
grad_g1_x, g1_x = gradient(g1, x)
@test abs(g1_x - gx) <= (1 + abs(gx))*1e-12
@test norm(grad_gx - grad_g1_x, Inf) <= 1e-12

call_test(g2, x)
grad_g2_x, g2_x = gradient(g2, x)
@test abs(g2_x - gx) <= (1 + abs(gx))*1e-12
@test norm(grad_gx - grad_g2_x, Inf) <= 1e-12

call_test(g3, x)
grad_g3_x, g3_x = gradient(g3, x)
@test abs(g3_x - gx) <= (1 + abs(gx))*1e-12
@test norm(grad_gx - grad_g3_x, Inf) <= 1e-12

y1, gy1 = prox_test(g1, x, 1.0)
y2, gy2 = prox_test(g2, x, 1.0)
@test abs(gy1 - gy2) <= (1 + abs(gy1))*1e-12
@test norm(y1 - y2) <= (1 + norm(y1))*1e-12

y3, gy3 = prox_test(g3, x, 1.0)
@test abs(gy2 - gy3) <= (1 + abs(gy2))*1e-12
@test norm(y2 - y3) <= (1 + norm(y2))*1e-12

# IndSOC composed with [I, I, I]

f = IndSOC()
A = [Matrix{Float64}(I, 3, 3) Matrix{Float64}(I, 3, 3) Matrix{Float64}(I, 3, 3)]

g = Precompose(f, A, 3.0)

x = [0.4, 0.2, 0.4, 0.5, 0.3, 0.3, 0.6, 0.4, 0.2]

@test g(x) == 0.0
call_test(g, x)
y, gy = prox_test(g, x, 1.0)

x = [0.1, 0.2, 0.4, 0.2, 0.3, 0.3, 0.3, 0.4, 0.2]

@test g(x) == Inf
call_test(g, x)
y, gy = prox_test(g, x, 1.0)

# ElasticNet composed with [diag, diag, diag]

f = ElasticNet()
d = 1.0:10.0
n = length(d)
A = [Diagonal(d) Diagonal(d) Diagonal(d)]

g = Precompose(f, A, 3*(Array(d).^2))

x = randn(3*n)

call_test(g, x)
y, gy = prox_test(g, x, 1.0)
