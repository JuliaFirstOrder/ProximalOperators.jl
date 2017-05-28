println("testing deep Array operations")

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
