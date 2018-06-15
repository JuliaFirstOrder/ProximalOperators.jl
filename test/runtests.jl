using ProximalOperators
using Base.Test

srand(0)

TOL_ASSERT = 1e-12

# measures time and returns the result of the call method
function call_test(f, x)
  print("* call        : ");
  try
    @time fx = f(x)
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
  TOL_ASSERT_PROX = ProximalOperators.is_prox_accurate(f) ? 1e-12 : 1e-6
  print("* prox        : "); @time yf, fy = prox(f, x, gamma)
  print("* prox!       : "); yf_prealloc = deepcopy(x); @time fy_prealloc = prox!(yf_prealloc, f, x, gamma)
  print("* prox_naive  : "); @time y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)
  @test ProximalOperators.deepmaxabs(yf_prealloc .- yf)/(1 + ProximalOperators.deepmaxabs(yf)) <= TOL_ASSERT_PROX
  @test ProximalOperators.deepmaxabs(y_naive .- yf)/(1 + ProximalOperators.deepmaxabs(yf)) <= TOL_ASSERT_PROX
  if ProximalOperators.is_set(f)
    @test fy_prealloc == 0
  end
  @test fy_prealloc == fy || abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT_PROX
  @test fy_naive == fy || abs(fy_naive - fy)/(1+abs(fy_naive)) <= TOL_ASSERT_PROX
  if !ProximalOperators.is_set(f) || ProximalOperators.is_prox_accurate(f)
    try
      f_at_y = f(yf)
      @test f_at_y == fy || abs(fy - f_at_y)/(1+abs(fy)) <= TOL_ASSERT_PROX
    catch e
    end
  end
  return yf, fy
end

# test predicates consistency
# i.e., that more specific properties imply less specific ones
# e.g., the indicator of a subspace is the indicator of a set in particular
function predicates_test(f)
  # is_quadratic => is_generalized_quadratic
  @test !ProximalOperators.is_quadratic(f) ||
    ProximalOperators.is_generalized_quadratic(f)
  # is_(singleton || cone || affine) => is_set
  @test !(ProximalOperators.is_singleton(f) ||
    ProximalOperators.is_cone(f) ||
    ProximalOperators.is_affine(f)) ||
    ProximalOperators.is_set(f)
  # is_strongly_convex => is_convex
  @test !ProximalOperators.is_strongly_convex(f) ||
    ProximalOperators.is_convex(f)
end

@testset "ProximalOperators" begin

@testset "Utilities" begin
  include("test_deep.jl")
  include("test_symmetricpacked.jl")
end

@testset "Functions" begin
  include("test_huberLoss.jl")
  include("test_indAffine.jl")
  include("test_indPolyhedral.jl")
  include("test_leastSquares.jl")
  include("test_logisticLoss.jl")
  include("test_quadratic.jl")
  include("test_calls.jl")
  include("test_graph.jl")
end

@testset "Gradients" begin
  include("test_gradients.jl")
end

@testset "Calculus rules" begin
  include("test_calculus.jl")
  include("test_moreauEnvelope.jl")
  include("test_precompose.jl")
  include("test_postcompose.jl")
  include("test_regularize.jl")
  include("test_separableSum.jl")
  include("test_slicedSeparableSum.jl")
  include("test_sum.jl")
end

@testset "Equivalences" begin
  include("test_equivalences.jl")
end

@testset "Conditions" begin
  include("test_condition.jl")
end

@testset "Hardcoded" begin
  include("test_results.jl")
end

@testset "Demos" begin
  include("test_demos.jl")
end

end
