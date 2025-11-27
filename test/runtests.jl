using Test

using ProximalOperators
using ProximalOperators: ArrayOrTuple
using ProximalCore:
    is_proximable,
    is_separable,
    is_convex,
    is_singleton_indicator,
    is_cone_indicator,
    is_affine_indicator,
    is_set_indicator,
    is_smooth,
    is_quadratic,
    is_generalized_quadratic,
    is_strongly_convex,
    is_positively_homogeneous,
    is_support

using Aqua

function call_test(f, x::ArrayOrTuple{R}) where R <: Real
    try
        fx = @inferred f(x)
        @test typeof(fx) == R
        return fx
    catch e
        if !isa(e, MethodError)
            return nothing
        end
    end
end

Base.zero(xs::Tuple) = Base.zero.(xs)

# tests equality of the results of prox, prox! and prox_naive
function prox_test(f, x::ArrayOrTuple{R}, gamma=1) where R <: Real
    y, fy = @inferred prox(f, x, gamma)

    @test typeof(fy) == R

    y_prealloc = zero(x)
    fy_prealloc = prox!(y_prealloc, f, x, gamma)

    @test typeof(fy_prealloc) == R

    y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)

    @test typeof(fy_naive) == R
    
    rtol = if is_proximable(f) sqrt(eps(R)) else 1e-4 end

    if is_convex(f)
        @test all(isapprox.(y_prealloc, y, rtol=rtol, atol=100*eps(R)))
        @test all(isapprox.(y_naive, y, rtol=rtol, atol=100*eps(R)))
        if is_set_indicator(f)
            @test fy_prealloc == 0
        end
        @test isapprox(fy_prealloc, fy, rtol=rtol, atol=100*eps(R))
        @test isapprox(fy_naive, fy, rtol=rtol, atol=100*eps(R))
    end

    if !is_set_indicator(f) || is_proximable(f)
        f_at_y = call_test(f, y)
        if f_at_y !== nothing
            @test isapprox(f_at_y, fy, rtol=rtol, atol=100*eps(R))
        end
    end

    return y, fy
end

# tests equality of the results of prox, prox! and prox_naive
function gradient_test(f, x::ArrayOrTuple{R}, gamma=R(1)) where R <: Real
    grad_fx, fx = gradient(f, x)
    @test typeof(fx) == R
    return grad_fx, fx
end

# test predicates consistency
# i.e., that more specific properties imply less specific ones
# e.g., the indicator of a subspace is the indicator of a set in particular
function predicates_test(f)
    preds = [
        is_convex,
        is_strongly_convex,
        is_generalized_quadratic,
        is_quadratic,
        is_smooth,
        is_singleton_indicator,
        is_cone_indicator,
        is_affine_indicator,
        is_set_indicator,
        is_positively_homogeneous,
        is_support,
    ]

    for pred in preds
        # check that the value of the predicate can be inferred
        @inferred (arg -> Val(pred(arg)))(f)
    end

    # quadratic => generalized_quadratic && smooth
    @test !is_quadratic(f) || (is_generalized_quadratic(f) && is_smooth(f))
    # (singleton || cone || affine) => set
    @test !(is_singleton_indicator(f) || is_cone_indicator(f) || is_affine_indicator(f)) || is_set_indicator(f)
    # cone => positively homogeneous
    @test !is_cone_indicator(f) || is_positively_homogeneous(f)
    # (convex && positively homogeneous) <=> (convex && support)
    @test (is_convex(f) && is_positively_homogeneous(f)) == (is_convex(f) && is_support(f))
    # strongly_convex => convex
    @test !is_strongly_convex(f) || is_convex(f)
end

@testset "Aqua" begin
    Aqua.test_all(ProximalOperators; ambiguities=false)
end

@testset "Utilities" begin
    include("test_symmetricpacked.jl")
end

@testset "Functions" begin
    include("test_cubeNormL2.jl")
    include("test_huberLoss.jl")
    include("test_indAffine.jl")
    include("test_leastSquares.jl")
    include("test_logisticLoss.jl")
    include("test_quadratic.jl")
    include("test_linear.jl")
    include("test_indHyperslab.jl")
    include("test_graph.jl")
    include("test_normL1plusL2.jl")
end

include("test_calls.jl")
include("test_indPolyhedral.jl")

@testset "Gradients" begin
    include("test_gradients.jl")
end

@testset "Calculus rules" begin
    include("test_calculus.jl")
    include("test_epicompose.jl")
    include("test_moreauEnvelope.jl")
    include("test_precompose.jl")
    include("test_pointwiseMinimum.jl")
    include("test_postcompose.jl")
    include("test_regularize.jl")
    include("test_separableSum.jl")
    include("test_slicedSeparableSum.jl")
    include("test_sum.jl")
end

@testset "Equivalences" begin
    include("test_equivalences.jl")
end

include("test_optimality_conditions.jl")

@testset "Hardcoded" begin
    include("test_results.jl")
end

@testset "Demos" begin
    include("../demos/lasso.jl")
    include("../demos/rpca.jl")
end
