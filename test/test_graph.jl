## Test IndGraph

srand(0)
m, n = (50, 100)
sm = n + div(n, 10)

function test_against_IndAffine(f, A, cd)
    m, n = size(A)
    if m > n
        return 0.0  # no variant for skinny case for now
    end

    T = typeof(A[1,1])

    ## TODO: remove when IndAffine will be revised
    if T <: Complex{Float64}
        return 0.0
    end

    B = ifelse(issparse(A), [A -speye(m)],  [A -eye(m)])
    # INIT IndAffine for the case
    faff = IndAffine(B, zeros(T, m))

    xy_aff, fx_aff = prox(faff, cd)
    @test faff(xy_aff) == 0.0

    # INIT testing function
    xy, fx = prox(f, cd)
    @test f(xy) == 0.0

    # test against IndAFfine
    if T <: Complex
        return 0.0# currently complex case has different mappings, though both valid
    else
        @test ProximalOperators.deepmaxabs(xy - xy_aff) <= TOL_ASSERT
    end
    return 0.0
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

##### argument split
    c = view(x, 1:f.n)
    d = view(x, f.n + 1:f.n + f.m)
    ax = zeros(c)
    ay = zeros(d)

##### just call f
    fx = call_test(f, x)

##### compute prox with default gamma
    y, fy = prox_test(f, x)

##### compute prox with random gamma
    gam = 5*rand()
    y, fy = prox_test(f, x, gam)

##### test calls to prox! with more signatures
    prox!(ax, ay, f, c, d)
    @test f(ax, ay) <= TOL_ASSERT
    ax_naive, ay_naive, fv_naive = ProximalOperators.prox_naive(f, c, d, 1.0)
    @test f(ax_naive, ay_naive) <= TOL_ASSERT


    prox!((ax, ay), f, (c, d))
    @test f((ax, ay)) <= TOL_ASSERT
    axy_naive, fv_naive = ProximalOperators.prox_naive(f, (c, d))
    @test f(axy_naive) <= TOL_ASSERT

##### test against IndAffine
    test_against_IndAffine(f, params[1], x)
  end
end
