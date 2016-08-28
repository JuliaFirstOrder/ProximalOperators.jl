using Prox
using Base.Test

stuff = [
          Dict( "constr" => DistL2,
                "wrong"  => [ (IndBallL2(), -rand()) ],
                "params" => [ (IndSOC(),), (IndNonnegative(), rand()) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => SqrDistL2,
                "wrong"  => [ (IndBallL2(), -rand()) ],
                "params" => [ (IndSimplex(),), (IndNonnegative(), rand()) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => ElasticNet,
                "wrong"  => [ (-rand()), (-rand(), -rand()), (-rand(), rand()), (rand(), -rand()) ],
                "params" => [ (), (rand(),), (rand(), rand()) ],
                "args"   => ( randn(10), randn(10), randn(10) )
              ),

          Dict( "constr" => IndAffine,
                "params" => [ (randn(10), randn()), (randn(10,20), randn(10)) ],
                "args"   => ( randn(10), randn(20) ),
              ),

          Dict( "constr" => IndBallInf,
                "wrong"  => [ (-rand(),), ],
                "params" => [ (rand(),), ],
                "args"   => ( randn(20), )
              ),

          Dict( "constr" => IndBallL0,
                "wrong"  => [ (-4,), ],
                "params" => [ (5,), (5,) ],
                "args"   => ( randn(25), randn(35), )
              ),

          Dict( "constr" => IndBallL2,
                "wrong"  => [ (-rand(),), ],
                "params" => [ (rand(),), ],
                "args"   => ( randn(10), )
              ),

          Dict( "constr" => IndBallRank,
                "wrong"  => [ (-2,), ],
                "params" => [ (1+Int(round(10*rand())),), ],
                "args"   => ( randn(20, 50), )
              ),

          Dict( "constr" => IndBox,
                "wrong"  => [ (+1, -1), ],
                "params" => [ (-rand(),+rand(10)), (-rand(10),+rand()), (-rand(),+rand()) ],
                "args"   => ( randn(10), randn(10), randn(20) )
              ),

          Dict( "constr" => IndHalfspace,
                "params" => [ (rand(10),rand()), (rand(10),rand()), (rand(10),rand()), (rand(10),rand()) ],
                "args"   => ( randn(10), randn(10), randn(10), randn(10) ),
              ),

          Dict( "constr" => IndNonnegative,
                "params" => [ (), ],
                "args"   => ( randn(10), )
              ),

          Dict( "constr" => IndSimplex,
                "params" => [ (), ],
                "args"   => ( randn(10), )
              ),

          Dict( "constr" => IndSOC,
                "params" => [ (), (), (), (), (), (), () ],
                "args"   => ( [rand(), -rand()], [-rand(), rand()], rand(3), rand(5), randn(10), randn(20), randn(30) )
              ),

          Dict( "constr" => NormL0,
                "params" => [ (), (rand(),) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => NormL1,
                "params" => [ (), (rand(),), (rand(20),) ],
                "args"   => ( randn(10), randn(10), randn(20) )
              ),

          Dict( "constr" => NormL2,
                "params" => [ (), (rand(),) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => NormL21,
                "params" => [ (), (rand(),), (rand(),1), (rand(),2) ],
                "args"   => ( randn(10,20), randn(10,20), randn(10,20), randn(10,20) )
              ),

          Dict( "constr" => SqrNormL2,
                "params" => [ (), (rand(),), (rand(20),) ],
                "args"   => ( randn(10), randn(10), randn(20) )
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
    print("* call : "); @time fx = f(x)
    print("* prox : "); @time y, fy = prox(f, x)
    gam = 5*rand()
    print("* prox : "); @time y, fy = prox(f, x, gam)
    f_at_y = f(y)
    @test abs(fy - f_at_y)/(1+abs(fy)) <= 1e-14
  end
end
