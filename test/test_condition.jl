# test whether prox satisfies some conditions

stuff = [
  Dict( "constr" => LeastSquares,
        "params" => ( (randn(20, 10), randn(20)), (randn(15, 40), randn(15), rand()), (rand(Complex{Float64}, 15, 40), rand(Complex{Float64}, 15), rand()), (sprandn(100,1000,0.05), randn(100), rand()) ),
        "args"   => ( randn(10), randn(40), rand(Complex{Float64}, 40), randn(1000) ),
        "gammas" => ( rand(), rand(), rand(), rand() ),
        # test first order optimality condition of prox
        "test"   => (f, x, gamma, y) -> norm(y + gamma*f.lambda*(f.A'*(f.A*y-f.b)) - x) <= 1e-10
      ),

  Dict( "constr" => IndSimplex,
        "params" => ( (), (2,), (5.0,), (rand(),) ),
        "args"   => ( randn(10), randn(10), randn(10), randn(10) ),
        "gammas" => ( rand(), rand(), rand(), rand() ),
        # test y belonging to the simplex
        "test"   => (f, x, gamma, y) -> all(y .>= 0.0) && abs(sum(y)-f.a) <= (1+f.a)*1e-12
      ),

  Dict( "constr" => IndBallL1,
        "params" => ( (), (1.7,), (5.0,), (rand(), ) ),
        "args"   => ( randn(10), randn(10), randn(10), randn(10) ),
        "gammas" => ( rand(), rand(), rand(), rand() ),
        # test y belonging to the L1 ball
        "test"   => (f, x, gamma, y) -> vecnorm(y, 1) <= (1+1e-12)*f.r
      ),

  Dict( "constr" => HuberLoss,
        "params" => ( (), (rand(),), (rand(), rand()), (rand(), rand()) ),
        "args"   => ( randn(10), randn(8, 10), randn(20), rand(Complex{Float64}, 12, 15) ),
        "gammas" => ( rand(), rand(), rand(), rand() ),
        # test optimality condition of prox
        "test"   => (f, x, gamma, y) -> isapprox((x-y)/gamma, (vecnorm(y) <= f.rho ? f.mu*y : f.rho*f.mu*y/vecnorm(y)))
      ),
]

for i = 1:length(stuff)
  constr = stuff[i]["constr"]
  params = stuff[i]["params"]
  args = stuff[i]["args"]
  gammas = stuff[i]["gammas"]
  test = stuff[i]["test"]
  for i = 1:length(params)
    println("----------------------------------------------------------")
    println(constr)
    f = constr(params[i]...)
    println(f)
    y, fy = prox(f, args[i], gammas[i])
    @test test(f, args[i], gammas[i], y)
  end
end
