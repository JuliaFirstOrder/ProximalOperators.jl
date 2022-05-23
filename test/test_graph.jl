using LinearAlgebra
using SparseArrays
using Random

Random.seed!(0)

## Test IndGraph

m, n = (50, 100)
sm = n + div(n, 10)

function test_against_IndAffine(f, A, cd)
    m, n = size(A)
    if m > n
        return  # no variant for skinny case for now
    end

    T = eltype(A)

    ## TODO: remove when IndAffine will be revised
    if T <: Complex
        return
    end

    B = ifelse(issparse(A), [A -SparseMatrixCSC{T}(I, m, m)],  [A -Matrix{T}(I, m, m)])
    # INIT IndAffine for the case
    faff = IndAffine(B, zeros(T, m))

    xy_aff, fx_aff = prox(faff, cd)
    @test faff(xy_aff) == 0.0

    # INIT testing function
    xy, fx = prox(f, cd)
    @test f(xy) == 0.0

    # test against IndAFfine
    if T <: Complex
        return  # currently complex case has different mappings, though both valid
    else
        @test xy ≈ xy_aff
    end
    
    return
end

## First, do common tests
stuff = [
        Dict(
            "constr" => IndGraph,
            "params" => (
                    (sprand(m, n, 0.2),),
                    (sprand(Complex{Float64}, m, n, 0.2),),
                    (rand(m, n),),
                    (rand(Complex{Float64}, m, n),),
                    (rand(sm, n),),
                    (rand(Complex{Float64}, sm, n),),
            ),
            "args"   => (
                    randn(m + n),
                    randn(m + n)+im * randn(m + n),
                    randn(m + n),
                    randn(m + n)+im * randn(m + n),
                    randn(sm + n),
                    randn(sm + n)+im * randn(sm + n)
            )
      ),
]

for i in eachindex(stuff)
  constr = stuff[i]["constr"]

  if haskey(stuff[i], "wrong")
    for j in eachindex(stuff[i]["wrong"])
      wrong = stuff[i]["wrong"][j]
      @test_throws ErrorException constr(wrong...)
    end
  end

  for j in eachindex(stuff[i]["params"])
    params = stuff[i]["params"][j]
    x      = stuff[i]["args"][j]
    f = constr(params...)

    predicates_test(f)

##### argument split
    c = view(x, 1:f.n)
    d = view(x, f.n + 1:f.n + f.m)
    ax = zero(c)
    ay = zero(d)

##### just call f
    fx = call_test(f, x)

##### compute prox with default gamma
    y, fy = prox_test(f, x)

##### compute prox with random gamma
    gam = 5*rand()
    y, fy = prox_test(f, x, gam)

##### test calls to prox! with more signatures
    prox!(ax, ay, f, c, d)
    @test f(ax, ay) ≈ 0
    ax_naive, ay_naive, fv_naive = ProximalOperators.prox_naive(f, c, d, 1)
    @test f(ax_naive, ay_naive) ≈ 0


    prox!((ax, ay), f, (c, d))
    @test f((ax, ay)) ≈ 0
    axy_naive, fv_naive = ProximalOperators.prox_naive(f, (c, d), 1)
    @test f(axy_naive) ≈ 0

##### test against IndAffine
    test_against_IndAffine(f, params[1], x)
  end
end
