using Prox

stuff = [
          Dict( "constr" => DistL2,
                "params" => [ (IndSOC(10),), (IndNonnegative(), rand()) ]
              ),

          Dict( "constr" => SqrDistL2,
                "params" => [ (IndSimplex(),), (IndNonnegative(), rand()) ]
              ),

          Dict( "constr" => ElasticNet,
                "params" => [ (), (rand(),), (rand(), rand()) ]
              ),

          Dict( "constr" => IndAffine,
                "params" => [ (randn(10), randn()), (randn(10,20), randn(10)) ]
              ),

          Dict( "constr" => IndBallInf,
                "params" => [ (rand(),), ]
              ),

          Dict( "constr" => IndBallL0,
                "params" => [ (1+Int(round(10*rand())),), ]
              ),

          Dict( "constr" => IndBallL2,
                "params" => [ (rand(),), ]
              ),

          Dict( "constr" => IndBallRank,
                "params" => [ (1+Int(round(10*rand())),), ]
              ),

          Dict( "constr" => IndBox,
                "params" => [ (-rand(),+rand(10)), (-rand(10),+rand()), (-rand(),+rand()) ]
              ),

          Dict( "constr" => IndHalfspace,
                "params" => [ (rand(10),rand()), ]
              ),

          Dict( "constr" => IndNonnegative,
                "params" => [ (), ]
              ),

          Dict( "constr" => IndSimplex,
                "params" => [ (), ]
              ),

          Dict( "constr" => IndSOC,
                "params" => [ (1+Int(round(10*rand())),), ]
              ),

          Dict( "constr" => NormL0,
                "params" => [ (), (rand(),) ]
              ),

          Dict( "constr" => NormL1,
                "params" => [ (), (rand(),) ]
              ),

          Dict( "constr" => NormL2,
                "params" => [ (), (rand(),) ]
              ),

          Dict( "constr" => NormL21,
                "params" => [ (), (rand(),) ]
              ),

          Dict( "constr" => SqrNormL2,
                "params" => [ (), (rand(),) ]
              )
        ]

for i = 1:length(stuff)
  constr = stuff[i]["constr"]
  for j = 1:length(stuff[i]["params"])
    params = stuff[i]["params"][j]
    f = constr(params...)
  end
end
