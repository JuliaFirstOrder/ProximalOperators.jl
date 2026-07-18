using LinearAlgebra
using SparseArrays
using Random
using Test

Random.seed!(0)

# Simply test the call to functions and their prox

test_cases_spec = [
    Dict(
        "constr" => DistL2,
        "wrong"  => [
            (IndBallL2(), -rand()),
        ],
        "right" => [
            ( (IndSOC(),), randn(Float32, 10) ),
            ( (IndSOC(),), randn(Float64, 10) ),
            ( (IndNonnegative(), rand()),  randn(Float64, 10) ),
            ( (IndZero(),), randn(Float64, 10) ),
            ( (IndBox(-1, 1),), randn(Float32, 10) ),
            ( (IndBox(-1, 1),), randn(Float64, 10) ),
        ],
    ),

    Dict(
        "constr" => SqrDistL2,
        "wrong"  => [
            (IndBallL2(), -rand()),
        ],
        "right" => [
            ( (IndSimplex(),), randn(Float32, 10) ),
            ( (IndSimplex(),), randn(Float64, 10) ),
            ( (IndUnitSimplex(),), randn(Float32, 10) ),
            ( (IndUnitSimplex(),), randn(Float64, 10) ),
            ( (IndNonnegative(), rand()), randn(Float64, 10) ),
            ( (IndZero(),), randn(Float64, 10) ),
            ( (IndBox(-1, 1),), randn(Float32, 10) ),
            ( (IndBox(-1, 1),), randn(Float64, 10) ),
        ],
    ),

    Dict(
        "constr" => ElasticNet,
        "wrong"  => [
            (-rand()),
            (-rand(), -rand()),
            (-rand(), rand()),
            (rand(), -rand())
        ],
        "right" => [
            ( (), randn(Float64, 10) ),
            ( (rand(Float32),), randn(Float32, 10) ),
            ( (rand(Float64), rand(Float64)), randn(Float64, 10) ),
            ( (rand(Float64), rand(Float64)), rand(Complex{Float64}, 20) ),
        ],
    ),

    Dict(
        "constr" => HingeLoss,
        "wrong"  => [
            (randn(10), -rand()),
        ],
        "right" => [
            ( (Int.(sign.(randn(10))), ), randn(Float32, 10) ),
            ( (Int.(sign.(randn(10))), ), randn(Float64, 10) ),
            ( (Int.(sign.(randn(20))), 0.1f0 + rand(Float32)), randn(Float32, 20) ),
            ( (Int.(sign.(randn(20))), 0.1e0 + rand(Float64)), randn(Float64, 20) ),
        ],
    ),

    Dict(
        "constr" => IndAffine,
        "right" => [
            ( (randn(10), randn()), randn(Float32, 10) ),
            ( (randn(10, 20), randn(10)), randn(20) ),
            ( (sprand(50,100, 0.1), randn(50)), randn(Float32, 100) ),
            ( (sprand(Complex{Float64}, 50, 100, 0.1), randn(50)+im*randn(50)), randn(100)+im*randn(100) ),
        ],
    ),

    Dict(
        "constr" => IndBallLinf,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (rand(Float32),), randn(Float32, 10) ),
            ( (rand(Float64),), randn(Float64, 10) ),
        ],
    ),

    Dict(
        "constr" => IndBallL0,
        "wrong"  => [
            (-4,),
        ],
        "right" => [
            ( (5,), randn(25) ),
            ( (10, ), randn(Float32, 5, 5) ),
            ( (5, ), randn(Complex{Float64}, 15) ),
        ],
    ),

    Dict(
        "constr" => IndBallL1,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), [0.1, -0.2, 0.3, -0.39] ),
            ( (), Complex{Float64}[0.1, -0.2, 0.3, -0.39] ),
            ( (), rand(Float32, 5) ),
            ( (), rand(Float64, 5) ),
            ( (), rand(Complex{Float32}, 5) ),
            ( (Float32(3),), randn(Float32, 5) ),
            ( (Float64(5),), randn(Float64, 2, 3) ),
            ( (Float64(1),), randn(Complex{Float64}, 5) ),
        ],
    ),

    Dict(
        "constr" => IndBallL2,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), rand(Float32, 5) ),
            ( (), rand(Float64, 5) ),
            ( (), rand(Complex{Float32}, 5) ),
            ( (Float32(3),), randn(Float32, 5) ),
            ( (Float64(5),), randn(Float64, 2, 3) ),
            ( (Float64(1),), randn(Complex{Float64}, 5) ),
        ],
    ),

    Dict(
        "constr" => IndBallRank,
        "wrong"  => [
            (-2,),
        ],
        "right" => [
            ( (1+Int(round(10*rand())),), randn(20, 50) ),
            ( (10+Int(round(5*rand())),), rand(30, 8)*rand(8,70) ),
            ( (10+Int(round(5*rand())),), randn(Float32, 5, 8) ),
            ( (10+Int(round(5*rand())),), rand(Complex{Float64}, 20, 50) ),
            ( (10+Int(round(5*rand())),), rand(Complex{Float32}, 5, 8) ),
        ],
    ),

    Dict(
        "constr" => IndBox,
        "wrong"  => [
            (+1, -1),
        ],
        "right" => [
            ( (-rand(Float32), +rand(Float32, 10)), randn(Float32, 10) ),
            ( (-rand(Float32, 10), +rand(Float32)), randn(Float32, 10) ),
            ( (-rand(Float64), +rand(Float64)), randn(Float64, 20) ),
            ( (-rand(Float64, 30), +rand(Float64, 30)), randn(Float64, 30) ),
        ],
    ),

    Dict(
        "constr" => IndExpPrimal,
        "right" => [
            ( (), randn(Float32, 3) ),
            ( (), randn(Float64, 3) ),
        ],
    ),

    Dict(
        "constr" => IndExpDual,
        "right" => [
            ( (), randn(Float32, 3) ),
            ( (), randn(Float64, 3) ),
        ],
    ),

    Dict(
        "constr" => IndFree,
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 2, 3) ),
        ],
    ),

    Dict(
        "constr" => IndGraph,
        "right" => [
            ( (sprand(50, 100, 0.2),), randn(150) ),
            ( (sprand(Complex{Float64}, 50, 100, 0.2),), randn(150)+im*randn(150) ),
            ( (rand(50, 100),), randn(150) ),
            ( (rand(Complex{Float64}, 50, 100),), randn(150)+im*randn(150) ),
            ( (rand(55, 50),), randn(105) ),
            ( (rand(Complex{Float64}, 55, 50),), randn(105)+im*randn(105) ),
        ],
    ),

    Dict(
        "constr" => IndPoint,
        "right" => [
            ( (), randn(5) ),
            ( (randn(Float32, 5),), randn(Float32, 5) ),
            ( (randn(Float64, 5),), randn(Float32, 5) ),
            ( (randn(Float64, 5),), randn(Float64, 5) ),
            ( (randn(Complex{Float32}, 5),), randn(Complex{Float32}, 5) ),
            ( (randn(Complex{Float64}, 5),), randn(Complex{Float32}, 5) ),
            ( (randn(Complex{Float64}, 5),), randn(Complex{Float64}, 5) ),
        ],
    ),

    Dict(
        "constr" => IndStiefel,
        "right" => [
            ( (), randn(Float32, 5, 3) ),
            ( (), randn(Float64, 5, 3) ),
            ( (), randn(Complex{Float32}, 5, 3) ),
            ( (), randn(Complex{Float64}, 5, 3) ),
        ],
    ),

    Dict(
        "constr" => IndZero,
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float32, 2, 3) ),
            ( (), randn(Float64, 5) ),
            ( (), randn(Float64, 2, 3) ),
            ( (), randn(Complex{Float32}, 5) ),
            ( (), randn(Complex{Float32}, 2, 3) ),
        ],
    ),

    Dict(
        "constr" => IndHalfspace,
        "right" => [
            ( (-ones(5), -1.0), -rand(5) ),
            ( (-ones(5), 1.0), rand(5) ),
        ],
    ),

    Dict(
        "constr" => IndNonnegative,
        "right" => [
            ( (), randn(10) ),
            ( (), randn(3, 5) ),
        ],
    ),

    Dict(
        "constr" => IndNonpositive,
        "right" => [
            ( (), randn(10) ),
            ( (), randn(3, 5) ),
        ],
    ),

    Dict(
        "constr" => IndSimplex,
        "wrong"  => [
            (-rand(),)
        ],
        "right" => [
            ( (), randn(10) ),
            ( (), randn(3, 5) ),
            ( (5.0,), randn(10) ),
        ],
    ),

    Dict(
        "constr" => IndSOC,
        "right" => [
            ( (), [rand(), -rand()] ),
            ( (), [-rand(), rand()] ),
            ( (), rand(5) ),
        ],
    ),

    Dict(
        "constr" => IndRotatedSOC,
        "right" => [
            ( (), [rand(), -rand()] ),
            ( (), [-rand(), rand()] ),
            ( (), rand(5) ),
        ],
    ),

    Dict(
        "constr" => IndSphereL2,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (rand(),), randn(10) ),
            ( (sqrt(20),), randn(20) ),
            ( (1,), 1e-3*randn(10) ),
            ( (1,), randn(Complex{Float64}, 10) ),
        ],
    ),

    Dict(
        "constr" => IndPSD,
        "right" => [
            ( (), Symmetric(randn(Float32, 5, 5)) ),
            ( (), Symmetric(randn(Float64, 10, 10)) ),
            ( (), Hermitian(randn(Complex{Float32}, 5, 5)) ),
            ( (), Hermitian(randn(Complex{Float64}, 10, 10)) ),
            ( (), randn(Float32, 5, 5) ),
            ( (), randn(Float64, 10, 10) ),
            ( (), randn(Complex{Float32}, 5, 5) ),
            ( (), randn(Complex{Float64}, 10, 10) ),
            ( (), randn(15) ),
        ],
    ),

    Dict(
        "constr" => LogBarrier,
        "wrong"  => [
            (1.0, 0.0, -rand()),
        ],
        "right" => [
            ( (), rand(Float32, 5) ),
            ( (), rand(Float64, 5) ),
            ( (rand(Float32),), rand(Float32, 5) ),
            ( (rand(Float64), rand(Float64)), rand(Float64, 5) ),
            ( (rand(Float32), rand(Float32), rand(Float32)), rand(Float32, 5) ),
        ],
    ),

    Dict(
        "constr" => NormL0,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 5) ),
            ( (rand(Float32)), randn(Complex{Float32}, 5) ),
        ],
    ),

    Dict(
        "constr" => NormL1,
        "wrong"  => [
            (-rand(),),
            (-rand(10),)
        ],
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 5) ),
            ( (), randn(Complex{Float32}, 5) ),
            ( (rand(Float32),), randn(Float32, 5) ),
            ( (rand(Float64, 5),), randn(Float64, 5) ),
            ( (rand(Float64, 5),), rand(Complex{Float64}, 5) ),
            ( (rand(Float32),), rand(Complex{Float32}, 5) ),
        ],
    ),

    Dict(
        "constr" => NormL2,
        "right" => [
            ( (), randn(Float32, 5), ),
            ( (), randn(Float64, 5), ),
            ( (), randn(Complex{Float32}, 5), ),
            ( (rand(Float32),), randn(Float32, 5), ),
            ( (rand(Float64),), randn(Complex{Float64}, 5), ),
        ],
    ),

    Dict(
        "constr" => NormL1plusL2,
        "wrong"  => [
            (-rand(), rand()),
            (-rand(10),),
            (-rand(10), rand()),
        ],
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 5) ),
            ( (), randn(Complex{Float32}, 5) ),
            ( (rand(Float32),), randn(Float32, 5) ),
            ( (rand(Float64, 5),), randn(Float64, 5) ),
            ( (rand(Float64, 5),), rand(Complex{Float64}, 5) ),
            ( (rand(Float32),), rand(Complex{Float32}, 5) ),
            ( (rand(Float32), rand(Float32)), randn(Float32, 5) ),
            ( (rand(Float64, 5), rand(Float64)), randn(Float64, 5) ),
            ( (rand(Float64, 5), rand(Float64)), rand(Complex{Float64}, 5) ),
            ( (rand(Float32), rand(Float32)), rand(Complex{Float32}, 5) ),
        ],
    ),

    Dict(
        "constr" => NormL21,
        "right" => [
            ( (), randn(Float32, 3, 5) ),
            ( (), randn(Float64, 3, 5) ),
            ( (), randn(Complex{Float32}, 3, 5) ),
            ( (rand(Float32),), randn(Float32, 3, 5) ),
            ( (rand(Float64),), randn(Float64, 3, 5) ),
            ( (rand(Float32), 1), randn(Complex{Float32}, 3, 5) ),
            ( (rand(Float64), 2), randn(Complex{Float64}, 3, 5) ),
        ],
    ),

    Dict(
        "constr" => NormLinf,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 5) ),
            ( (), randn(Complex{Float32}, 5) ),
            ( (rand(Float32),), randn(Float32, 5) ),
            ( (rand(Float32),), rand(Complex{Float32}, 5) ),
        ],
    ),

    Dict(
        "constr" => NuclearNorm,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), rand(Float32, 5, 5) ),
            ( (), rand(Float64, 3, 5) ),
            ( (), rand(Complex{Float32}, 5, 5) ),
            ( (rand(Float32),), rand(Float32, 5, 5) ),
            ( (rand(Float64),), rand(Float64, 3, 5) ),
            ( (rand(Float32),), rand(Complex{Float32}, 5, 5) ),
            ( (sqrt(10e0),), rand(Complex{Float64}, 3, 5) ),
            ( (sqrt(10f0),), rand(Complex{Float32}, 5, 3) ),
        ],
    ),

    Dict(
        "constr" => SqrNormL2,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), randn(Float32, 10) ),
            ( (), randn(Float64, 10) ),
            ( (rand(Float32),), randn(Float32, 10) ),
            ( (rand(Float64),), randn(Float64, 10) ),
            ( (Float64[1, 1, 1, 1, 0],), randn(Float64, 5) ),
            ( (rand(Float32, 20),), randn(Float32, 20) ),
            ( (rand(Float64, 20),), randn(Float64, 20) ),
            ( (rand(30),), rand(Complex{Float64}, 30) ),
            ( (rand(),), rand(Complex{Float64}, 50) ),
        ],
    ),

    Dict(
        "constr" => LeastSquares,
        "wrong"  => [
            (randn(3, 5), randn(4), rand()),
        ],
        "right" => [
            ( (randn(10, 25), randn(10)), randn(25) ),
            ( (randn(40, 13), randn(40), rand()), randn(13) ),
            ( (rand(Complex{Float64}, 25, 10), rand(Complex{Float64}, 25)), rand(Complex{Float64}, 10) ),
            ( (sprandn(10, 100, 0.15), randn(10), rand()), randn(100) ),
        ],
    ),

    # Dict(
    #     "constr" => SumLargest,
    #     "wrong"  => [
    #         (-1,), (0,), (1, -2.0), (2, 0.0), (2, -rand())
    #     ],
    #     "right" => [
    #         ( (), randn(10) ),
    #         ( (1,), randn(20) ),
    #         ( (5, 3.2), randn(5,10) ),
    #         ( (3, rand()), randn(8,17) ),
    #     ],
    # ),

    Dict(
        "constr" => Maximum,
        "wrong"  => [
            (-rand(),),
        ],
        "right" => [
            ( (), randn(Float32, 10) ),
            ( (), randn(Float64, 10) ),
            ( (rand(Float64),), randn(Float64, 20) ),
            ( (), randn(Float32, 5,10) ),
            ( (rand(Float32)), randn(Float32, 8,17) ),
        ],
    ),

    Dict(
        "constr" => IndBinary,
        "right" => [
            ( (), randn(10) ),
            ( (randn(), randn()), randn(10) ),
            ( (randn(), randn(10)), randn(10) ),
            ( (randn(10), randn()), randn(10) ),
            ( (randn(10), randn(10)), randn(10) ),
            ( (randn(), randn(5,5)), randn(5,5) ),
        ],
    ),

    Dict(
        "constr" => Regularize,
        "wrong"  => [
            (NormL1(), -rand()),
        ],
        "right" => [
            ( (NormL1(), rand()), randn(10) ),
            ( (NormL1(rand()), rand()), randn(5,10) ),
            ( (NormL1(), rand(), randn(20)), randn(20) ),
            ( (NormL1(rand()), rand(), randn(20)), randn(20) ),
        ],
    ),

    Dict(
        "constr" => Tilt,
        "right" => [
            ( (LeastSquares(randn(20, 10), randn(20)), randn(10)), randn(10) ),
        ],
    ),

    Dict(
        "constr" => HuberLoss,
        "wrong"  => [
            (-rand(), ),
            (rand(), -rand()),
            (-rand(), rand()),
            (-rand(), -rand())
        ],
        "right" => [
            ( (), randn(Float32, 10) ),
            ( (), randn(Float64, 10) ),
            ( (rand(Float32), ), randn(Float32, 5, 8) ),
            ( (rand(Float64), ), randn(Float64, 5, 8) ),
            ( (rand(Float64), rand(Float64)), randn(Float64, 20) ),
            ( (rand(Float64), rand(Float64)), rand(Complex{Float64}, 8, 12) ),
        ],
    ),

    Dict(
        "constr" => SumPositive,
        "right" => [
            ( (), randn(Float32, 5) ),
            ( (), randn(Float64, 5) ),
            ( (), randn(Float32, 3, 5) ),
            ( (), randn(Float64, 3, 5) ),
        ],
    ),

    Dict(
        "constr" => SeparableSum,
        "right" => [
            ( ((NormL2(2.0), NormL1(1.5), NormL2(0.5)), ), (randn(5), randn(15), randn(10)) ),
        ],
    ),

    Dict(
        "constr" => SlicedSeparableSum,
        "right" => [
            ( ((NormL2(2.0), NormL1(1.5), NormL2(0.5)), ((1:5,), (6:20,), (21:30,))), randn(30) ),
        ],
    ),
]

@testset "$(spec["constr"])" for spec in test_cases_spec
    constr = spec["constr"]

    if haskey(spec, "wrong")
        for wrong in spec["wrong"]
            @test_throws ErrorException constr(wrong...)
        end
    end

    for right in spec["right"]
        params, x = right
        f = constr(params...)

        predicates_test(f)

        ##### just call f
        fx = call_test(f, x)

        ##### compute prox with default gamma
        y, fy = prox_test(f, x)

        ##### compute prox with random gamma
        T = if typeof(x) <: Array
            eltype(x)
        elseif typeof(x) <: Tuple
            eltype(x[1])
        else
            Float64
        end
        gam = 5*rand(real(T))
        y, fy = prox_test(f, x, gam)

        ##### compute prox with multiple random gammas
        if is_separable(f)
            gam = real(T)(0.5) .+ 2 .* rand(real(T), size(x))
            y, fy = prox_test(f, x, gam)
        end

    end
end
