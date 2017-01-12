
lambda = rand()
rho = rand()
g = Regularize(NormL1(lambda),rho)
show(g)
println()

x = randn(10)
y,f = prox(g,x)
y2,f2 = prox(ElasticNet(lambda,rho),x)

@test norm(f-f2)<TOL_ASSERT
@test norm(y-y2)<TOL_ASSERT
