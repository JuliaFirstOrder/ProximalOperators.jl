@printf("\nTesting deep operations\n")

x = ([2.0, 3.0], [4.0, 5.0, 6.0], [1.0 2.0 3.0; 4.0 5.0 6.0])

lengths_x = (2, 3, 6)
deeplength_x = 11
deepvecnorm_x = 13.45362404707371

@test length.(x) == lengths_x
@test ProximalOperators.deeplength(x) == deeplength_x

y = ProximalOperators.deepsimilar(x)

@test ProximalOperators.deeplength(y) == deeplength_x
@test length.(y) == lengths_x

ProximalOperators.deepcopy!(y, x)

@test y == x
@test ProximalOperators.deepvecnorm(x) ≈ deepvecnorm_x
@test ProximalOperators.deepvecdot(x, y) ≈ deepvecnorm_x^2
@test ProximalOperators.deepmaxabs(x .- y) == 0

t1 = ProximalOperators.deepzeros((Float32, Float64), ((3, ), (4, )) )

t1 = (randn(20), randn(20))
t2 = (randn(20), randn(20))
t3 = ProximalOperators.deepsimilar(t1)
ProximalOperators.deepaxpy!(t3, t1, 0.5, t2)
t4 = (t1[1]+0.5*t2[1],t1[2]+0.5*t2[2])
@test max(norm(t4[1]-t3[1]),norm(t4[2]-t3[2])) <= 1e-12
