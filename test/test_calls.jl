# Simply test the call to functions and their prox

TOL_ASSERT = 1e-12

stuff = [
  Dict( "constr" => DistL2,
        "wrong"  => ( (IndBallL2(), -rand()), ),
        "params" => ( (IndSOC(),), (IndNonnegative(), rand()) ),
        "args"   => ( randn(10), randn(10) )
      ),

  Dict( "constr" => SqrDistL2,
        "wrong"  => ( (IndBallL2(), -rand()), ),
        "params" => ( (IndSimplex(),), (IndNonnegative(), rand()) ),
        "args"   => ( randn(10), randn(10) )
      ),

  Dict( "constr" => ElasticNet,
        "wrong"  => ( (-rand()), (-rand(), -rand()), (-rand(), rand()), (rand(), -rand()) ),
        "params" => ( (), (rand(),), (rand(), rand()), (rand(), rand()) ),
        "args"   => ( randn(10), randn(10), randn(10), rand(Complex{Float64},20) )
      ),

  Dict( "constr" => HingeLoss,
        "wrong"  => ( (randn(10), -rand()), ),
        "params" => ( (sign(randn(10)), ), (sign(randn(20)), 0.1+rand()) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndAffine,
        "params" => ( (randn(10), randn()), (randn(10,20), randn(10)) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndBallInf,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), ),
        "args"   => ( randn(20), )
      ),

  Dict( "constr" => IndBallL0,
        "wrong"  => ( (-4,), ),
        "params" => ( (5,), (5,) ),
        "args"   => ( randn(25), randn(35), )
      ),

  Dict( "constr" => IndBallL1,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (3.0,), (0.4,), (rand() + 0.1,) ),
        "args"   => ( rand(15), randn(25), randn(35), randn(60) )
      ),

  Dict( "constr" => IndBallL2,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), (sqrt(20)) ),
        "args"   => ( randn(10), randn(20), )
      ),

  Dict( "constr" => IndBallRank,
        "wrong"  => ( (-2,), ),
        "params" => ( (1+Int(round(10*rand())),), (10+Int(round(5*rand())),), (10+Int(round(5*rand())),), (10+Int(round(5*rand())),), (10+Int(round(5*rand())),) ),
        "args"   => ( randn(20, 50), rand(30, 8)*rand(8,70), randn(5, 8), rand(Complex{Float64}, 20, 50), rand(Complex{Float64}, 5, 8) )
      ),

  Dict( "constr" => IndBox,
        "wrong"  => ( (+1, -1), ),
        "params" => ( (-rand(),+rand(10)), (-rand(10),+rand()), (-rand(),+rand()), (-rand(30),+rand(30)) ),
        "args"   => ( randn(10), randn(10), randn(20), randn(30) )
      ),

  Dict( "constr" => IndHalfspace,
        "params" => ( (rand(10),rand()), (rand(20),rand()), (rand(30),rand()), (rand(50),rand()) ),
        "args"   => ( randn(10), randn(20), randn(30), randn(50) ),
      ),

  Dict( "constr" => IndNonnegative,
        "params" => ( (), ),
        "args"   => ( randn(10), )
      ),

  Dict( "constr" => IndNonpositive,
        "params" => ( (), ),
        "args"   => ( randn(10), )
      ),

  Dict( "constr" => IndSimplex,
        "wrong"  => ( (-rand(),) ),
        "params" => ( (), ),
        "args"   => ( randn(20), )
      ),

  Dict( "constr" => IndSOC,
        "params" => ( (), (), (), (), (), (), () ),
        "args"   => ( [rand(), -rand()], [-rand(), rand()], rand(3), rand(5), randn(10), randn(20), randn(30) )
      ),

  Dict( "constr" => IndSphereL2,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), (sqrt(20),) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndPSD,
        "params" => ( (), () ),
        "args"   => ( Symmetric(randn(5,5)), Symmetric(rand(20,20)) ),
      ),

  Dict( "constr" => LogBarrier,
        "wrong"  => ( (1.0, 0.0, -rand()), ),
        "params" => ( (), (rand(),), (rand(), rand()), (rand(), rand(), rand()) ),
        "args"   => ( rand(10), rand(20), rand(30), rand(50) )
      ),

  Dict( "constr" => NormL0,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (rand(),) ),
        "args"   => ( randn(10), randn(10) )
      ),

  Dict( "constr" => NormL1,
        "wrong"  => ( (-rand(),), (-rand(10),) ),
        "params" => ( (), (rand(),), (rand(20),), (rand(30),), (rand(),) ),
        "args"   => ( randn(10), randn(10), randn(20), rand(Complex{Float64},30), rand(Complex{Float64}, 50) )
      ),

  Dict( "constr" => NormL2,
        "params" => ( (), (rand(),) ),
        "args"   => ( randn(10), randn(10) )
      ),

  Dict( "constr" => NormL21,
        "params" => ( (), (rand(),), (rand(),1), (rand(),2) ),
        "args"   => ( randn(10,20), randn(10,20), randn(10,20), randn(10,20) )
      ),

  Dict( "constr" => NuclearNorm,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), (rand(),), (rand(),), (sqrt(10),) ),
        "args"   => ( rand(10,10), rand(100,25), rand(Complex{Float64},20,20), rand(Complex{Float64},30,20) )
      ),

  Dict( "constr" => SqrNormL2,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (rand(),), (rand(20),), (rand(30),), (rand(),) ),
        "args"   => ( randn(10), randn(10), randn(20), rand(Complex{Float64}, 30), rand(Complex{Float64}, 50) )
      )
]

for i = 1:length(stuff)
  constr = stuff[i]["constr"]

  if haskey(stuff[i], "wrong")
    for j = 1:length(stuff[i]["wrong"])
      wrong = stuff[i]["wrong"][j]
      @test_throws ErrorException constr(wrong...)
    end
  end

  for j = 1:length(stuff[i]["params"])
    println("----------------------------------------------------------")
    println(constr)
    params = stuff[i]["params"][j]
    x      = stuff[i]["args"][j]
    f = constr(params...)
    println(f)

##### just call f
    print("* call                        : "); @time fx = f(x)

##### compute prox with default gamma
    print("* prox                        : "); @time y, fy = prox(f, x)
    print("* prox! (preallocated output) : "); y_prealloc = copy(x); @time fy_prealloc = prox!(f, x, y_prealloc)
    print("* prox! (in place)            : "); y_inplace = copy(x); @time fy_inplace = prox!(f, y_inplace)

    # compare the three versions
    @test vecnorm(y_prealloc - y)/(1+vecnorm(y)) <= TOL_ASSERT
    @test abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT
    @test vecnorm(y_inplace - y)/(1+vecnorm(y)) <= TOL_ASSERT
    @test abs(fy_inplace - fy)/(1+abs(fy_inplace)) <= TOL_ASSERT

    # compare with naive implementation
    y_naive, fy_naive = Prox.prox_naive(f, x)
    @test vecnorm(y_naive - y)/(1+vecnorm(y_naive)) <= TOL_ASSERT
    @test abs(fy_naive - fy)/(1+abs(fy_naive)) <= TOL_ASSERT
    f_at_y = f(y)
    @test abs(fy - f_at_y)/(1+abs(fy)) <= TOL_ASSERT

##### compute prox with random gamma
    gam = 5*rand()
    print("* prox                        : "); @time y, fy = prox(f, x, gam)
    print("* prox! (preallocated output) : "); y_prealloc = copy(x); @time fy_prealloc = prox!(f, x, y_prealloc, gam)
    print("* prox! (in place)            : "); y_inplace = copy(x); @time fy_inplace = prox!(f, y_inplace, gam)

    # compare the three versions
    @test vecnorm(y_prealloc - y)/(1+vecnorm(y)) <= TOL_ASSERT
    @test abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT
    @test vecnorm(y_inplace - y)/(1+vecnorm(y)) <= TOL_ASSERT
    @test abs(fy_inplace - fy)/(1+abs(fy_inplace)) <= TOL_ASSERT

    # compare with naive implementation
    y_naive, fy_naive = Prox.prox_naive(f, x, gam)
    @test vecnorm(y_naive - y)/(1+vecnorm(y_naive)) <= TOL_ASSERT
    @test abs(fy_naive - fy)/(1+abs(fy_naive)) <= TOL_ASSERT
    f_at_y = f(y)
    @test abs(fy - f_at_y)/(1+abs(fy)) <= TOL_ASSERT

  end
end
