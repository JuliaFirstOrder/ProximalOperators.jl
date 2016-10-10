# Test equivalence of functions and prox mappings by means of calculus rules

TOL_ASSERT = 1e-12

stuff = [
  Dict( "funcs"  => (lambda -> (NormL1(lambda), Conjugate(IndBallInf(lambda))))(0.1+10.0*rand()),
        "args"   => ( 5.0*sign(randn(10)) + 5.0*randn(10),
                      5.0*sign(randn(20)) + 5.0*randn(20) ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),

  Dict( "funcs"  => (lambda -> (IndBallInf(lambda), Conjugate(NormL1(lambda))))(0.1+10.0*rand()),
        "args"   => ( 5.0*sign(randn(10)) + 5.0*randn(10),
                      5.0*sign(randn(20)) + 5.0*randn(20) ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),

  Dict( "funcs"  => (lambda -> (NormL1(lambda), Conjugate(IndBox(-lambda,lambda))))(0.1+10.0*rand(30)),
        "args"   => ( 5.0*sign(randn(30)) + 5.0*randn(30), ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),

  Dict( "funcs"  => (lambda -> (IndBox(-lambda,lambda), Conjugate(NormL1(lambda))))(0.1+10.0*rand(30)),
        "args"   => ( 5.0*sign(randn(30)) + 5.0*randn(30), ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),

  Dict( "funcs"  => (lambda -> (NormL2(lambda), Conjugate(IndBallL2(lambda))))(0.1+10.0*rand()),
        "args"   => ( 5.0*sign(randn(10)) + 5.0*randn(10),
                      5.0*sign(randn(20)) + 5.0*randn(20) ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),

  Dict( "funcs"  => (lambda -> (IndBallL2(lambda), Conjugate(NormL2(lambda))))(0.1+10.0*rand()),
        "args"   => ( 5.0*sign(randn(10)) + 5.0*randn(10),
                      5.0*sign(randn(20)) + 5.0*randn(20) ),
        "gammas" => ( 0.5+rand(), 0.5+rand() )
      ),
]

for i = 1:length(stuff)

  f = stuff[i]["funcs"][1]
  g = stuff[i]["funcs"][2]

  println(string("testing ", rpad(typeof(f), 25, " "), " vs. ", typeof(g)))

  for j = 1:length(stuff[i]["args"])
    x = stuff[i]["args"][j]
    gamma = stuff[i]["gammas"][j]
    yf, vf = prox(f, x, gamma)
    yg, vg = prox(g, x, gamma)
    @test vecnorm(yf-yg, Inf)/(1+norm(yf, Inf)) <= TOL_ASSERT
  end

end
