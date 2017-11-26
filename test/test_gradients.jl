
stuff = [
  Dict( "f"      => ElasticNet(2.0, 3.0),
        "x"      => [-2., -1., 0., 1., 2., 3.],
        "∇f(x)"  => 2*[-1., -1., 0., 1., 1., 1.] + 3*[-2., -1., 0., 1., 2., 3.],
      ),
  Dict( "f"      => NormL1(2.0),
        "x"      => [-2., -1., 0., 1., 2., 3.],
        "∇f(x)"  => 2*[-1., -1., 0., 1., 1., 1.],
      ),
  Dict( "f"      => NormL2(2.0),
        "x"      => [-1., 0., 2.],
        "∇f(x)"  => 2/sqrt(5)*[-1., 0., 2.],
      ),
  Dict( "f"      => NormL2(2.0),
        "x"      => [-1., 0., 2.],
        "∇f(x)"  => 2/sqrt(5)*[-1., 0., 2.],
      ),
  Dict( "f"      => NormL2(2.0),
        "x"      => [0., 0., 0.],
        "∇f(x)"  => [0., 0., 0.],
      ),
  # Dict( "f"      => NormL21(),
  #       "x"      => ,
  #       "∇f(x)"  => ,
  #     ),
  Dict( "f"      => NormLinf(2.),
        "x"      => [-2., -1., 0., 1., 2., 3.],
        "∇f(x)"  => 2*[0., 0., 0., 0., 0., 1.],
      ),
  Dict( "f"      => NormLinf(2.),
        "x"      => [-4., -1., 0., 1., 2., 3.],
        "∇f(x)"  => 2*[-1., 0., 0., 0., 0., 0.],
      ),
  # Dict( "f"      => NuclearNorm(),
  #       "x"      => ,
  #       "∇f(x)"  => ,
  #     ),
  Dict( "f"      => SqrNormL2(2.0),
        "x"      => [-2., -1., 0., 1., 2., 3.],
        "∇f(x)"  => 2*[-2., -1., 0., 1., 2., 3.],
      ),
  Dict( "f"      => HingeLoss([1., 2., 1., 2., 1.], 2.0),
        "x"      => [-2., -1., 0., 2., 3.],
        "∇f(x)"  => -2*[1., 2., 1.,  0., 0.],
      ),
  Dict( "f"      => HuberLoss(2., 3.),
        "x"      => [-1., 0.5],
        "∇f(x)"  => 3.*[-1., 0.5],
      ),
  Dict( "f"      => HuberLoss(2., 3.),
        "x"      => [-2., 1.5],
        "∇f(x)"  => [-4.8, 3.6], # 3*2*[-2., 1.5]/norm([-2., 1.5]),
      ),
  Dict( "f"      => Linear([-1.,2.,3.]),
        "x"      => [1., 5., 3.14],
        "∇f(x)"  => [-1.,2.,3.],
      ),
  Dict( "f"      => LogBarrier(2.0, 1.0, 1.5),
        "x"      => [1.0, 2.0, 3.0],
        "∇f(x)"  => -1.5*2.0./(2.0*[1.0, 2.0, 3.0].+1.0),# -μ*a*/(a*x+b)
      ),
  Dict( "f"      => LeastSquares([1.0 2.0; 3.0 4.0], [0.1, 0.2], 2.0, iterative=false),
        "x"      => [-1., 2.],
        "∇f(x)"  => [34.6, 50.], #λ*(A'A*x-A'b),
      ),
  Dict( "f"      => LeastSquares([1.0 2.0; 3.0 4.0], [0.1, 0.2], 2.0, iterative=true),
        "x"      => [-1., 2.],
        "∇f(x)"  => [34.6, 50.], #λ*(A'A*x-A'b),
      ),
  Dict( "f"      => Quadratic([2. -1.; -1. 2.], [0.1, 0.2], iterative=false),
        "x"      => [3., 4.],
        "∇f(x)"  => [2.1, 5.2], #[2. -1.; -1. 2.]*[3., 4.]+[0.1, 0.2],
      ),
  Dict( "f"      => Quadratic([2. -1.; -1. 2.], [0.1, 0.2], iterative=true),
        "x"      => [3., 4.],
        "∇f(x)"  => [2.1, 5.2], #[2. -1.; -1. 2.]*[3., 4.]+[0.1, 0.2],
      ),
  Dict( "f"      => SumPositive(),
        "x"      => [-1., 1., 2.],
        "∇f(x)"  => [0., 1., 1.],
      ),
  # Dict( "f"      => Maximum(2.0),
  #       "x"      => [-4., 2., 3.],
  #       "∇f(x)"  => 2*[0., 0., 1.],
  #     ),
  Dict( "f"      => DistL2(IndZero(),2.0),
        "x"      => [1., -2],
        "∇f(x)"  => 2*[1., -2]./norm([1., -2]),
      ),
  Dict( "f"      => DistL2(IndBallL2(1.0),2.0),
        "x"      => [2., -2],
        "∇f(x)"  => [sqrt(2), -sqrt(2)],
      ),
  Dict( "f"      => SqrDistL2(IndZero(),2.0),
        "x"      => [1., -2],
        "∇f(x)"  => 2*[1., -2],
      ),
  Dict( "f"      => SqrDistL2(IndBallL2(1.0),2.0),
        "x"      => [2., -2],
        "∇f(x)"  => 2.0*([2., -2]-[1/sqrt(2), -1/(sqrt(2))]),
      ),
  Dict( "f"      => LogisticLoss([1.0, -1.0, 1.0, -1.0, 1.0], 1.5),
        "x"      => [-1.0, -2.0, 3.0, 2.0, 1.0],
        "∇f(x)"  => [-1.0965878679450072, 0.17880438303317633, -0.07113880976635019, 1.3211956169668235, -0.4034121320549927]
      ),
]

println("Testing Gradients")
for i = 1:length(stuff)

  println("----------------------------------------------------------")

  f = stuff[i]["f"]
  x = stuff[i]["x"]

  println("Gradient of ", string(typeof(f), " on ", typeof(x)))

  ref_∇f = stuff[i]["∇f(x)"]

  ref_fx = f(x)
  ∇f = similar(x)
  fx = gradient!(∇f, f, x)
  @test fx == ref_fx || abs(fx-ref_fx)/(1+abs(ref_fx)) <= TOL_ASSERT
  @test ∇f == ref_∇f || norm(∇f-ref_∇f)/(1+norm(ref_∇f)) <= TOL_ASSERT

  for j = 1:11
    #For initial point x and 10 other random points
    fx = gradient!(∇f, f, x)
    for k = 1:10
      # Test conditions in different directions
      if ProximalOperators.is_convex(f)
        # Test ∇f is subgradient
        d = randn(size(x))
        @test f(x+d) ≥ fx + dot(d, ∇f) - TOL_ASSERT
      else
        # Assume smooth function
        d = randn(size(x))
        d ./= norm(d).*sqrt(TOL_ASSERT)
        @test f(x+d) ≈ fx + dot(d, ∇f) atol=TOL_ASSERT
      end
    end
    x = rand(size(x))
  end
end
