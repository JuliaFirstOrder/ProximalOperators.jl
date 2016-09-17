using Prox
using Base.Test

f = NormL21()
X = randn(100,100)
Y = copy(X)
prox!(f,X)
Y, gY = Prox.prox_naive(f,Y)
@test vecnorm(X-Y)/(1+vecnorm(X)) <= 1e-14
