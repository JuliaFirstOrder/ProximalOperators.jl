using Test

using ProximalOperators

using ProximalOperators:
    ArrayOrTuple,
    is_prox_accurate,
    is_separable,
    is_convex,
    is_singleton,
    is_cone,
    is_affine,
    is_set,
    is_smooth,
    is_quadratic,
    is_generalized_quadratic,
    is_strongly_convex,
    is_positively_homogeneous,
    is_support

function call_test(f, x::ArrayOrTuple{R}) where R <: Real
    try
        fx = f(x)
        @test typeof(fx) == R
        return fx
    catch e
        if !isa(e, MethodError)
            return nothing
        end
    end
end

# tests equality of the results of prox, prox! and prox_naive
function prox_test(f, x::ArrayOrTuple{R}, gamma=R(1)) where R <: Real
    y, fy = prox(f, x, gamma)

    @test typeof(fy) == R

    y_prealloc = zero(x)
    fy_prealloc = prox!(y_prealloc, f, x, gamma)

    @test typeof(fy_prealloc) == R

    y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)

    @test typeof(fy_naive) == R
    
    rtol = if ProximalOperators.is_prox_accurate(f) sqrt(eps(R)) else 1e-4 end

    if ProximalOperators.is_convex(f)
        @test all(isapprox.(y_prealloc, y, rtol=rtol, atol=100*eps(R)))
        @test all(isapprox.(y_naive, y, rtol=rtol, atol=100*eps(R)))
        if ProximalOperators.is_set(f)
            @test fy_prealloc == 0
        end
        @test isapprox(fy_prealloc, fy, rtol=rtol, atol=100*eps(R))
        @test isapprox(fy_naive, fy, rtol=rtol, atol=100*eps(R))
    end

    if !ProximalOperators.is_set(f) || ProximalOperators.is_prox_accurate(f)
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
    # quadratic => generalized_quadratic && smooth
    @test !is_quadratic(f) || (is_generalized_quadratic(f) && is_smooth(f))
    # (singleton || cone || affine) => set
    @test !(is_singleton(f) || is_cone(f) || is_affine(f)) || is_set(f)
    # cone => positively homogeneous
    @test !is_cone(f) || is_positively_homogeneous(f)
    # (convex && positively homogeneous) <=> (convex && support)
    @test (is_convex(f) && is_positively_homogeneous(f)) == (is_convex(f) && is_support(f))
    # strongly_convex => convex
    @test !is_strongly_convex(f) || is_convex(f)
end

@testset "Sanity checks" begin
    @test isempty(detect_unbound_args(ProximalOperators))
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
