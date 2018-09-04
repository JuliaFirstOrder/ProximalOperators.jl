# test whether prox satisfies some conditions

using Random
using LinearAlgebra
using SparseArrays

Random.seed!(0)

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
        "test"   => (f, x, gamma, y) -> norm(y, 1) <= (1+1e-12)*f.r
      ),

  Dict( "constr" => HuberLoss,
        "params" => ( (), (rand(),), (rand(), rand()), (rand(), rand()) ),
        "args"   => ( randn(10), randn(8, 10), randn(20), rand(Complex{Float64}, 12, 15) ),
        "gammas" => ( rand(), rand(), rand(), rand() ),
        # test optimality condition of prox
        "test"   => (f, x, gamma, y) -> isapprox((x-y)/gamma, (norm(y) <= f.rho ? f.mu*y : f.rho*f.mu*y/norm(y)))
      ),

  Dict( "constr" => SqrHingeLoss,
        "params" => ( (randn(5),), (randn(5), 0.1+rand()), (randn(3, 5), 0.1+rand()) ),
        "args"   => ( randn(5), randn(5), randn(3, 5) ),
        "gammas" => ( 0.1+rand(), 0.1+rand(), 0.1+rand() ),
        # test optimality condition of prox
        "test"   => (f, x, gamma, z) -> isapprox((x-z)/gamma, -2 .* f.mu.*f.y.*max.(0, 1 .- f.y.*z))
      ),
]

for i = 1:length(stuff)
  constr = stuff[i]["constr"]
  params = stuff[i]["params"]
  args = stuff[i]["args"]
  gammas = stuff[i]["gammas"]
  test = stuff[i]["test"]
  for i = 1:length(params)
    # println("----------------------------------------------------------")
    # println(constr)
    f = constr(params[i]...)
    # println(f)
    y, fy = prox(f, args[i], gammas[i])
    @test test(f, args[i], gammas[i], y)
  end
end
