# Test other equivalences of prox operations which are not covered by calculus rules

using Random
using SparseArrays
using ProximalOperators
using Test

################################################################################
### testing consistency of simplex/L1 ball
################################################################################
# Inspired by Condat, "Fast projection onto the simplex and the l1 ball", Mathematical Programming, 158:575–585, 2016.
# See Prop. 2.1 there and following remarks.

@testset "IndSimplex/IndBallL1" begin

n = 20
N = 10

# projecting onto the L1 ball
for i = 1:N
  x = randn(n)
  r = 5*rand()
  f = IndSimplex(r)
  g = IndBallL1(r)

  y1, fy1 = prox(f, abs.(x))
  y1 = sign.(x).*y1
  y2, gy2 = prox(g, x)

  @test y1 ≈ y2
end

# projecting onto the simplex
for i = 1:N
  x = randn(n)
  r = 5*rand()
  f = IndSimplex(r)
  g = IndBallL1(r)

  y1, fy1 = prox(f, x)
  y2, gy2 = prox(g, x .- minimum(x) .+ r./n)

  @test y1 ≈ y2
end

end

################################################################################
### testing consistency of hinge loss/box indicator
################################################################################

@testset "HingeLoss/IndBox" begin

n = 20
N = 10

# test using Moreau identity: prox(f, x, gamma) = x - gamma*prox(f*, x/gamma, 1/gamma)
# and a couple of other calculus rules in the case b = ±ones(n)
#
# f(x) = max(0, x) is conjugate to h = IndBox(0,1)
# g(x) = HingeLoss(b, mu)(x) = mu*f(1-b.*x)
# prox(g, x, gamma) = (prox(f, 1-b.*x, mu*gamma) - 1)./(-b)
#   = x + mu*gamma*prox(h, (1-b.*x)/(mu*gamma), 1/(mu*gamma))./b

b = sign.(randn(n))
mu = 0.1+rand()
g = HingeLoss(b, mu)
h = IndBox(0, 1)

for i = 1:N
  x = randn(n)
  gamma = 0.1 + rand()
  y1, ~ = prox(g, x, gamma)
  z, ~ = prox(h, (1 .- b.*x)./(mu*gamma), 1/(mu*gamma))
  y2 = mu*gamma*(z./b) + x
  @test y1 ≈ y2
end

end

################################################################################
### testing regularize
################################################################################

@testset "Regularize/ElasticNet" begin

lambda = rand()
rho = rand()
g = Regularize(NormL1(lambda),rho)

x = randn(10)
y,f = prox(g,x)
y2,f2 = prox(ElasticNet(lambda,rho),x)

@test f ≈ f2
@test y ≈ y2

end

################################################################################
### testing IndAffine
################################################################################

@testset "IndAffine (sparse/dense)" begin

A = sprand(50,100, 0.1)
b = randn(50)

g1 = IndAffine(A, b)
g2 = IndAffine(Matrix(A), b)

x = randn(100)
y1, f1 = prox(g1, x)
y2, f2 = prox(g2, x)

@test f1 ≈ f2
@test y1 ≈ y2

end

################################################################################
### testing NormL1plusL2 reduces to L1/L2
################################################################################

@testset "NormL1plusL2 special case" begin

g = NormL1(1.)
# λ_2 = 0
f = NormL1plusL2(1., 0.)

x = randn(100)

y1, f1 = prox(g, x)
y2, f2 = prox(f, x)

@test f1 ≈ f2
@test y1 ≈ y2

end

@testset "NormL1plusL2 special case" begin

g = NormL2(1.)
# λ_1 = 0
f = NormL1plusL2(0., 1.)

x = randn(100)

y1, f1 = prox(g, x)
y2, f2 = prox(f, x)

@test f1 ≈ f2
@test y1 ≈ y2

end
