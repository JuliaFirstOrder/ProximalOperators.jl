# Simply test the call to functions and their prox

stuff = [
  Dict( "constr" => DistL2,
        "wrong"  => ( (IndBallL2(), -rand()), (IndBinary(),) ),
        "params" => ( (IndSOC(),), (IndNonnegative(), rand()), (IndZero(),), (IndZero(),), (IndBox(-1, 1),), (IndBox(-1, 1),) ),
        "args"   => ( randn(10), randn(10), 1e-1*randn(10), randn(10), 0.5*randn(10), randn(10) )
      ),

  Dict( "constr" => SqrDistL2,
        "wrong"  => ( (IndBallL2(), -rand()), (IndBinary(),) ),
        "params" => ( (IndSimplex(),), (IndNonnegative(), rand()), (IndZero(),), (IndZero(),), (IndBox(-1, 1),), (IndBox(-1, 1),) ),
        "args"   => ( randn(10), randn(10), 1e-1*randn(10), randn(10), 0.5*randn(10), randn(10) )
      ),

  Dict( "constr" => ElasticNet,
        "wrong"  => ( (-rand()), (-rand(), -rand()), (-rand(), rand()), (rand(), -rand()) ),
        "params" => ( (), (rand(),), (rand(), rand()), (rand(), rand()) ),
        "args"   => ( randn(10), randn(10), randn(10), rand(Complex{Float64},20) )
      ),

  Dict( "constr" => HingeLoss,
        "wrong"  => ( (randn(10), -rand()), ),
        "params" => ( (sign.(randn(10)), ), (sign.(randn(20)), 0.1+rand()) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndAffine,
        "params" => ( (randn(10), randn()), (randn(10,20), randn(10)) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndAffine,
        "params" => ( (sprand(50,100, 0.1), randn(50)), (sprand(Complex{Float64}, 50,100, 0.1), randn(50)+im*randn(50)), ),
        "args"   => ( randn(100), randn(100)+im*randn(100), )
      ),

  Dict( "constr" => IndBallLinf,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), ),
        "args"   => ( randn(20), )
      ),

  Dict( "constr" => IndBallL0,
        "wrong"  => ( (-4,), ),
        "params" => ( (5,), (5,), (10, ) ),
        "args"   => ( randn(25), randn(35), randn(10, 10) )
      ),

  Dict( "constr" => IndBallL1,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (3.0,), (0.4,), (rand() + 0.1,), (rand() + 0.1,) ),
        "args"   => ( rand(15), randn(25), randn(35), randn(60), randn(20,30) )
      ),

  Dict( "constr" => IndBallL2,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (rand(),), (sqrt(20),), (0.5,) ),
        "args"   => ( randn(10), randn(20), 0.1*ones(10) )
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

  Dict( "constr" => IndExpPrimal,
        "params" => ( (), (), () ),
        "args"   => ( randn(3), randn(3), randn(3) )
      ),

  Dict( "constr" => IndExpDual,
        "params" => ( (), (), () ),
        "args"   => ( randn(3), randn(3), randn(3) )
      ),

  Dict( "constr" => IndFree,
        "params" => ( (), (), () ),
        "args"   => ( randn(5), randn(10), randn(30) )
      ),
  Dict( "constr" => IndGraph,
        "params" => (
                (sprand(50,100, 0.2),),
                (sprand(Complex{Float64}, 50,100, 0.2),),
                (rand(50, 100),),
                (rand(Complex{Float64}, 50, 100),),
                (rand(55, 50),),
                (rand(Complex{Float64}, 55, 50),),
        ),
        "args"   => (
                randn(150),
                randn(150)+im*randn(150),
                randn(150),
                randn(150)+im*randn(150),
                randn(105),
                randn(105)+im*randn(105)
          )
      ),
  Dict( "constr" => IndPoint,
        "params" => ( (), (randn(20), ) ),
        "args"   => ( randn(10), randn(20) )
      ),

  Dict( "constr" => IndZero,
        "params" => ( (), (), () ),
        "args"   => ( randn(5), randn(10), randn(30) )
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
        "params" => ( (), (1.0+rand(),), (), (5.0,) ),
        "args"   => ( randn(20), randn(30), randn(10,10), randn(10) )
      ),

  Dict( "constr" => IndSOC,
        "params" => ( (), (), (), (), (), (), () ),
        "args"   => ( [rand(), -rand()], [-rand(), rand()], rand(3), rand(5), randn(10), randn(20), randn(30) )
      ),

  Dict( "constr" => IndRotatedSOC,
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

  Dict( "constr" => IndPSD,
        "params" => ( (), (), () ),
        "args"   => ( randn(6), randn(15), randn(55) )
      ),

  Dict( "constr" => IndZero,
        "params" => ( (), () ),
        "args"   => ( randn(10), randn(20) )
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

  Dict( "constr" => NormLinf,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (rand(),), (), (rand(),), (rand(),) ),
        "args"   => ( randn(10), randn(20), rand(Complex{Float64}, 10), rand(Complex{Float64}, 20), randn(10,10) )
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
      ),

  Dict( "constr" => LeastSquares,
        "wrong"  => ( (randn(3,5), randn(3), -rand()), (randn(3,5), randn(4), rand()) ),
        "params" => ( (randn(10,25), randn(10)), (randn(40,13), randn(40), rand()), (rand(Complex{Float64},25,10), rand(Complex{Float64},25)), (sprandn(100,1000,0.05), randn(100), rand()) ),
        "args"   => ( randn(25), randn(13), rand(Complex{Float64},10), randn(1000) )
      ),

  # Dict( "constr" => SumLargest,
  #       "wrong"  => ( (-1,), (0,), (1, -2.0), (2, 0.0), (2, -rand()) ),
  #       "params" => ( (), (1,), (5, 3.2), (3, rand()) ),
  #       "args"   => ( randn(10), randn(20), randn(5,10), randn(8,17) )
  #     ),

  Dict( "constr" => Maximum,
        "wrong"  => ( (-rand(),), ),
        "params" => ( (), (rand(),), (), (rand()) ),
        "args"   => ( randn(10), randn(20), randn(5,10), randn(8,17) )
      ),

  Dict( "constr" => IndBinary,
        "params" => ( (), (randn(), randn()), (randn(), randn(10)), (randn(10), randn()), (randn(10), randn(10)), (randn(), randn(5,5)) ),
        "args"   => ( randn(10), randn(10), randn(10), randn(10), randn(10), randn(5,5) )
      ),

  Dict( "constr" => Regularize,
        "wrong"  => ( (NormL1(), -rand()), ),
        "params" => ( (NormL1(), rand()), (NormL1(rand()), rand()), (NormL1(), rand(), randn(20)), (NormL1(rand()), rand(), randn(20)) ),
        "args"   => ( randn(10), randn(5,10), randn(20), randn(20) )
      ),

  Dict( "constr" => Tilt,
        "params" => ( (LeastSquares(randn(20, 10), randn(20)), randn(10)), ),
        "args"   => ( randn(10), )
      ),

  Dict( "constr" => HuberLoss,
        "wrong"  => ( (-rand(), ), (rand(), -rand()), (-rand(), rand()), (-rand(), -rand()) ),
        "params" => ( (), (rand(), ), (rand(), rand()), (rand(), rand()) ),
        "args"   => ( randn(10), randn(5, 8), randn(20), rand(Complex{Float64}, 8, 12) )
      ),

  Dict( "constr" => SumPositive,
        "params" => ( (), (), (), (), () ),
        "args"   => ( randn(3), randn(10), randn(12,19), randn(4,3), randn(17) )
      ),

  Dict( "constr" => SeparableSum,
        "params" => ( ((NormL2(2.0), NormL1(1.5), NormL2(0.5)), ), ),
        "args"   => ( (randn(5), randn(15), randn(10)), )
      ),

  Dict( "constr" => SlicedSeparableSum,
       "params" => ( ((NormL2(2.0), NormL1(1.5), NormL2(0.5)), ((1:5,), (6:20,), (21:30,))), ),
        "args"   => ( randn(30), )
      ),
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

    predicates_test(f)

##### just call f
    fx = call_test(f, x)

##### compute prox with default gamma
    y, fy = prox_test(f, x)

##### compute prox with random gamma
    gam = 5*rand()
    y, fy = prox_test(f, x, gam)

##### compute prox with multiple random gammas
    if ProximalOperators.is_separable(f)
      gam = 0.5+2*rand(size(x))
      y, fy = prox_test(f, x, gam)
    end

  end
end
