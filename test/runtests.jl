using ProximalOperators
using Base.Test

srand(0)

TOL_ASSERT = 1e-12

# measures time and returns the result of the call method
function call_test(f, x)
  print("* call        : ");
  try
    @time fx = f(x)
    @test typeof(fx) == eltype(real(x))
    return fx
  catch e
    if isa(e, MethodError)
      println("(not defined)")
    end
    return +Inf
  end
end

# measures time of the three possible calls to prox and prox!
# then tests equality of the results and returns them if they agree
function prox_test(f, x, gamma::Union{Real, AbstractArray}=1.0)
  print("* prox        : "); @time yf, fy = prox(f, x, gamma)
  print("* prox!       : "); yf_prealloc = copy(x); @time fy_prealloc = prox!(yf_prealloc, f, x, gamma)
  print("* prox_naive  : "); @time y_naive, fy_naive = ProximalOperators.prox_naive(f, x, gamma)
  @test typeof(fy) == eltype(real(x))
  @test vecnorm(yf_prealloc - yf, Inf)/(1+vecnorm(yf, Inf)) <= TOL_ASSERT
  @test vecnorm(y_naive - yf, Inf)/(1+vecnorm(yf, Inf)) <= TOL_ASSERT
  if ProximalOperators.is_cone(f)
    @test ProximalOperators.is_set(f)
  end
  if ProximalOperators.is_set(f)
    @test fy_prealloc == 0
  end
  if ProximalOperators.is_prox_accurate(f)
    @test fy_prealloc == fy || abs(fy_prealloc - fy)/(1+abs(fy)) <= TOL_ASSERT
    @test fy_naive == fy || abs(fy_naive - fy)/(1+abs(fy_naive)) <= TOL_ASSERT
    try
      f_at_y = f(yf)
      @test f_at_y == fy || abs(fy - f_at_y)/(1+abs(fy)) <= TOL_ASSERT
    catch e
    end
  end
  return yf, fy
end

println("*********************************************************************")
include("test_utilities.jl")
println("*********************************************************************")
include("test_calls.jl")
println("*********************************************************************")
include("test_calculus.jl")
println("*********************************************************************")
include("test_SeparableSum.jl")
include("test_SlicedSeparableSum.jl")
println("*********************************************************************")
include("test_equivalences.jl")
println("*********************************************************************")
include("test_condition.jl")
println("*********************************************************************")
include("test_results.jl")
println("*********************************************************************")
