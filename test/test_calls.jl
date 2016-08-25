using Prox
using Base.Test

stuff = [
          Dict( "constr" => DistL2,
                "params" => [ (IndSOC(10),), (IndNonnegative(), rand()) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => SqrDistL2,
                "params" => [ (IndSimplex(),), (IndNonnegative(), rand()) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => ElasticNet,
                "params" => [ (), (rand(),), (rand(), rand()) ],
                "args"   => ( randn(10), randn(10), randn(10) )
              ),

          Dict( "constr" => IndAffine,
                "params" => [ (randn(10), randn()), (randn(10,20), randn(10)) ],
                "args"   => ( randn(10), randn(20) )
              ),

          Dict( "constr" => IndBallInf,
                "params" => [ (rand(),), ],
                "args"   => ( randn(20), )
              ),

          Dict( "constr" => IndBallL0,
                "params" => [ (1+Int(round(10*rand())),), ],
                "args"   => ( randn(30), )
              ),

          Dict( "constr" => IndBallL2,
                "params" => [ (rand(),), ],
                "args"   => ( randn(10), )
              ),

          Dict( "constr" => IndBallRank,
                "params" => [ (1+Int(round(10*rand())),), ],
                "args"   => ( randn(20, 50), )
              ),

          Dict( "constr" => IndBox,
                "params" => [ (-rand(),+rand(10)), (-rand(10),+rand()), (-rand(),+rand()) ],
                "args"   => ( randn(10), randn(10), randn(20) )
              ),

          Dict( "constr" => IndHalfspace,
                "params" => [ (rand(10),rand()), ],
                "args"   => ( randn(10), )
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
                "params" => [ (1+Int(round(10*rand())),), ],
                "args"   => ( randn(10), )
              ),

          Dict( "constr" => NormL0,
                "params" => [ (), (rand(),) ],
                "args"   => ( randn(10), randn(10) )
              ),

          Dict( "constr" => NormL1,
                "params" => [ (), (rand(),) ],
                "args"   => ( randn(10), randn(10) )
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
                "params" => [ (), (rand(),) ],
                "args"   => ( randn(10), randn(10) )
              )
        ]

for i = 1:length(stuff)
  constr = stuff[i]["constr"]
  for j = 1:length(stuff[i]["params"])
    params = stuff[i]["params"][j]
    x      = stuff[i]["args"][j]
    f = constr(params...)
    print(constr, " (call) : "); @time fx = f(x)
    gamma = 5*rand()
    print(constr, " (prox) : "); @time y, fy = prox(f, gamma, x)
    f_at_y = f(y)
    @test abs(fy - f_at_y)/(1+abs(fy)) <= 1e-14
  end
end
