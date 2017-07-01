# Indicator of L1 norm ball composed with orthogonal matrix

f = IndBallL1()
A = randn(10, 10)
Q, ~ = qr(A)

@test vecnorm(Q'*Q - I) <= 1e-12
@test vecnorm(Q*Q' - I) <= 1e-12

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
Q, ~ = qr(A)

@test vecnorm(Q'*Q - I) <= 1e-12
@test vecnorm(Q*Q' - I) <= 1e-12

g = Precompose(f, Q, 1.0)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L1 norm composed with multiple of orthogonal matrix

f = NormL1()
A = randn(50, 50)
Q, ~ = qr(A)

@test vecnorm(Q'*Q - I) <= 1e-12
@test vecnorm(Q*Q' - I) <= 1e-12

g = Precompose(f, 3.0*Q, 9.0)

x = randn(50)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with multiple of orthogonal matrix

f = NormL2()
A = randn(500, 500)
Q, ~ = qr(A)

@test vecnorm(Q'*Q - I) <= 1e-12
@test vecnorm(Q*Q' - I) <= 1e-12

g = Precompose(f, 0.9*Q, 0.9^2)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with orthogonal matrix + translation

f = NormL2()
A = randn(500, 500)
b = randn(500)
Q, ~ = qr(A)

@test vecnorm(Q'*Q - I) <= 1e-12
@test vecnorm(Q*Q' - I) <= 1e-12

g = Precompose(f, Q, 1.0, -b)

x = randn(500)

call_test(g, x)
prox_test(g, x, 1.0)

# L2 norm composed with diagonal matrix + translation
# checking that Precompose and PrecomposeDiagonal agree

f = NormL2()
A = spdiagm(3.0*ones(500))
b = randn(500)

g1 = Precompose(f, A, 9.0, -b)
g2 = PrecomposeDiagonal(f, 3.0, -b)

x = randn(500)

call_test(g1, x)
y1, gy1 = prox_test(g1, x, 1.0)

call_test(g2, x)
y2, gy2 = prox_test(g2, x, 1.0)

@test abs(gy1 - gy2) <= (1 + abs(gy1))*1e-12
@test vecnorm(y1 - y2) <= (1 + vecnorm(y1))*1e-12

# Squared L2 norm composed with diagonal matrix + translation
# checking that Precompose and PrecomposeDiagonal agree
# checking that weighted squared L2 norm + Translate agrees too

f = SqrNormL2()
diagA = [rand(250); -rand(250)]
A = spdiagm(diagA)
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

y1, gy1 = prox_test(g1, x, 1.0)
y2, gy2 = prox_test(g2, x, 1.0)
@test abs(gy1 - gy2) <= (1 + abs(gy1))*1e-12
@test vecnorm(y1 - y2) <= (1 + vecnorm(y1))*1e-12

call_test(g3, x)
grad_g3_x, g3_x = gradient(g3, x)
@test abs(g3_x - gx) <= (1 + abs(gx))*1e-12
@test norm(grad_gx - grad_g3_x, Inf) <= 1e-12

y3, gy3 = prox_test(g3, x, 1.0)
@test abs(gy1 - gy3) <= (1 + abs(gy1))*1e-12
@test vecnorm(y1 - y3) <= (1 + vecnorm(y1))*1e-12
