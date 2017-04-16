println("testing deep Array operations")

x = [1.0, [2.0, 3.0], [[4.0, 5.0, 6.0], [1.0 2.0 3.0; 4.0 5.0 6.0]]]

lengths_x = [1, 2, 2]
deeplength_x = 12
deepvecnorm_x = 13.490737563232042

@test length.(x) == lengths_x
@test ProximalOperators.deeplength(x) == deeplength_x

y = ProximalOperators.deepsimilar(x)

@test ProximalOperators.deeplength(y) == deeplength_x
@test length.(y) == lengths_x

ProximalOperators.deepcopy!(y, x)

@test y == x
@test ProximalOperators.deepvecnorm(x) ≈ deepvecnorm_x
@test ProximalOperators.deepvecdot(x, y) ≈ deepvecnorm_x^2
@test ProximalOperators.deepmaxabs(x - y) == 0
