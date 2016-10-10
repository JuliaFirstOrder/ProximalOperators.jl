# Test other equivalences of prox operations which are not covered by calculus rules

ASSERT_REL_TOL = 1e-12

################################################################################
### testing consistency of simplex/L1 ball
################################################################################
# Inspired by Condat, "Fast projection onto the simplex and the l1 ball", Mathematical Programming, 158:575–585, 2016.
# See Prop. 2.1 there and following remarks.

println("testing indicator of simplex vs. L1 norm ball")

n = 20
N = 10

# projecting onto the L1 ball
for i = 1:N
  x = randn(n)
  r = 5*rand()
  f = IndSimplex(r)
  g = IndBallL1(r)

  y1, fy1 = prox(f, abs(x))
  y1 = sign(x).*y1
  y2, gy2 = prox(g, x)

  @test vecnorm(y1-y2,Inf)/(1+vecnorm(y1,Inf)) <= ASSERT_REL_TOL
end

# projecting onto the simplex
for i = 1:N
  x = randn(n)
  r = 5*rand()
  f = IndSimplex(r)
  g = IndBallL1(r)

  y1, fy1 = prox(f, x)
  y2, gy2 = prox(g, x-minimum(x)+r/n)

  @test vecnorm(y1-y2,Inf)/(1+vecnorm(y1,Inf)) <= ASSERT_REL_TOL
end

################################################################################
### testing consistency of hinge loss/box indicator
################################################################################

println("testing indicator of hinge loss vs. box indicator")

n = 20
N = 10

# test using Moreau identity: prox(f, x, gamma) = x - gamma*prox(f*, x/gamma, 1/gamma)
# and a couple of other calculus rules in the case b = ±ones(n)
#
# f(x) = max(0, x) is conjugate to h = IndBox(0,1)
# g(x) = HingeLoss(b, mu)(x) = mu*f(1-b.*x)
# prox(g, x, gamma) = (prox(f, 1-b.*x, mu*gamma) - 1)./(-b)
#   = x + mu*gamma*prox(h, (1-b.*x)/(mu*gamma), 1/(mu*gamma))./b

b = sign(randn(n))
mu = 0.1+rand()
g = HingeLoss(b, mu)
h = IndBox(0, 1)

for i = 1:N
  x = randn(n)
  gamma = 0.1 + rand()
  y1, ~ = prox(g, x, gamma)
  z, ~ = prox(h, (1-b.*x)/(mu*gamma), 1/(mu*gamma))
  y2 = mu*gamma*(z./b) + x
  @test vecnorm(y1-y2, Inf)/(1+norm(y1, Inf)) <= ASSERT_REL_TOL
end
