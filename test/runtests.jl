using Test
using Random
using Printf

using ProximalOperators
using ProximalOperators:
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
  is_strongly_convex

Random.seed!(0)

TOL_ASSERT = 1e-12

# measures time and returns the result of the call method
function call_test(f, x)
  # print("* call        : ");
  try
    # @time fx = f(x)
    fx = f(x)
    return fx
  catch e
    if isa(e, MethodError)
      println("(not defined)")
    end
    return +Inf
  end
end

# measures time of the calls to prox, prox! and prox_naive
# then tests equality of the results and returns them if they agree
function prox_test(f, x, gamma::Union{Real, AbstractArray}=1.0)
    # print("* prox        : "); @time y, fy = prox(f, x, gamma)
    y, fy = prox(f, x, gamma)
    # print("* prox!       : "); y_prealloc = similar(x); @time fy_prealloc = prox!(y_prealloc, f, x, gamma)
    y_prealloc = similar(x); fy_prealloc = prox!(y_prealloc, f, x, gamma)
    # print("* prox_naive  : "); @time y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)
    y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)
    if ProximalOperators.is_convex(f)
        # @test norm(yf_prealloc .- yf, Inf)/(1 + norm(yf, Inf)) <= TOL_ASSERT_PROX
        # @test norm(y_naive .- yf, Inf)/(1 + norm(yf, Inf)) <= TOL_ASSERT_PROX
        @test y_prealloc ≈ y
        @test y_naive ≈ y
        if ProximalOperators.is_set(f)
            @test fy_prealloc == 0
        end
        # @test fy_prealloc == fy || abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT_PROX
        # @test fy_naive == fy || abs(fy_naive - fy)/(1+abs(fy_naive)) <= TOL_ASSERT_PROX
        @test fy_prealloc ≈ fy
        @test fy_naive ≈ fy
    end
    if !ProximalOperators.is_set(f) || ProximalOperators.is_prox_accurate(f)
        try
            f_at_y = f(y)
            # @test f_at_y == fy || abs(fy - f_at_y)/(1+abs(fy)) <= TOL_ASSERT_PROX
            @test f_at_y ≈ fy
        catch e
        end
    end
    return y, fy
end

# test predicates consistency
# i.e., that more specific properties imply less specific ones
# e.g., the indicator of a subspace is the indicator of a set in particular
function predicates_test(f)
  # is_quadratic => is_(generalized_quadratic && smooth)
  @test !is_quadratic(f) || (is_generalized_quadratic(f) && is_smooth(f))
  # is_(singleton || cone || affine) => is_set
  @test !(is_singleton(f) || is_cone(f) || is_affine(f)) || is_set(f)
  # is_strongly_convex => is_convex
  @test !is_strongly_convex(f) || is_convex(f)
end

@testset "ProximalOperators" begin

@testset "Utilities" begin
  # include("test_deep.jl")
  include("test_symmetricpacked.jl")
end

@testset "Functions" begin
  # include("test_huberLoss.jl")
  # include("test_indAffine.jl")
  # include("test_indPolyhedral.jl")
  # include("test_leastSquares.jl")
  # include("test_logisticLoss.jl")
  # include("test_quadratic.jl")
  # include("test_linear.jl")
  # include("test_calls.jl")
  # include("test_graph.jl")
end

@testset "Gradients" begin
  # include("test_gradients.jl")
end

@testset "Calculus rules" begin
  # include("test_calculus.jl")
  include("test_moreauEnvelope.jl")
  # include("test_precompose.jl")
  include("test_postcompose.jl")
  include("test_regularize.jl")
  include("test_separableSum.jl")
  # include("test_slicedSeparableSum.jl")
  include("test_sum.jl")
end

@testset "Equivalences" begin
  # include("test_equivalences.jl")
end

@testset "Conditions" begin
  # include("test_condition.jl")
end

@testset "Hardcoded" begin
  # include("test_results.jl")
end

@testset "Demos" begin
  # include("test_demos.jl")
end

end
