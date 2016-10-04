# In several cases, the prox with respect to some function can be expressed
# in terms of the prox of some other function. One example of this is a norm
# vs. the indicator of the dual norm ball. More in general, Moreau identity.
#
# Still, in some of these cases it may be convenient, from the computational
# point of view, to write the prox of a function from scratch rather than
# reusing existing code. It is safe then to test the proximal mappings of
# closely related functions against each other.
#

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
