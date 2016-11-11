# test first order optimality condition of prox

stuff = [
  Dict( "constr" => LeastSquares,
        "params" => ( (randn(20, 10), randn(20)), (randn(15, 40), randn(15), rand()), (sprandn(100,1000,0.05), randn(100), rand()) ),
        "args"   => ( randn(10), randn(40), randn(1000) ),
        "gammas" => ( rand(), rand(), rand() ),
        "test"   => (f, x, gamma, y) -> norm(y + gamma*f.lambda*(f.A'*(f.A*y-f.b)) - x) <= 1e-10
      )
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
