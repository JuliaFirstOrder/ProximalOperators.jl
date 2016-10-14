using Prox
using Base.Test

TOL_ASSERT = 1e-12

# measures time and returns the result of the call method
function call_test(f, x)
  print("* call                        : "); @time fx = f(x)
  return fx
end

# measures time of the three possible calls to prox and prox!
# then tests equality of the results and returns them if they agree
function prox_test(f, x, gamma::Real=1.0)
  print("* prox                        : "); @time yf, fy = prox(f, x, gamma)
  print("* prox! (preallocated output) : "); yf_prealloc = copy(x); @time fy_prealloc = prox!(f, x, yf_prealloc, gamma)
  print("* prox! (in place)            : "); yf_inplace = copy(x); @time fy_inplace = prox!(f, yf_inplace, gamma)
  @test vecnorm(yf_prealloc - yf, Inf)/(1+vecnorm(yf, Inf)) <= TOL_ASSERT
  @test fy_prealloc == fy || abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT
  @test vecnorm(yf_inplace - yf, Inf)/(1+vecnorm(yf, Inf)) <= TOL_ASSERT
  @test fy_inplace == fy || abs(fy_inplace - fy)/(1+abs(fy_inplace)) <= TOL_ASSERT
  return yf, fy
end

include("test_calls.jl")
println("*********************************************************************")
include("test_calculus.jl")
println("*********************************************************************")
include("test_equivalences.jl")
println("*********************************************************************")
include("test_results.jl")
println("*********************************************************************")
