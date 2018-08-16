using Random

Random.seed!(123)

# smooth case

f1 = SqrNormL2()
f2 = Translate(SqrNormL2(2.5), randn(10))
f = Sum(f1, f2)

predicates_test(f)

@test ProximalOperators.is_quadratic(f) == true
@test ProximalOperators.is_strongly_convex(f) == true
@test ProximalOperators.is_set(f) == false

xtest = randn(10)

result = f1(xtest) + f2(xtest)
@test f(xtest) ≈ result

grad1, val1 = gradient(f1, xtest)
grad2, val2 = gradient(f2, xtest)

gradsum = randn(size(xtest))
valsum = gradient!(gradsum, f, xtest)
@test gradsum ≈ grad1 + grad2

# nonsmooth case

g1 = NormL2()
g2 = Translate(SqrNormL2(2.5), randn(10))
g = Sum(g1, g2)

predicates_test(g)

@test ProximalOperators.is_smooth(g) == false
@test ProximalOperators.is_strongly_convex(g) == true
@test ProximalOperators.is_set(g) == false

xtest = randn(10)

result = g1(xtest) + g2(xtest)
@test g(xtest) ≈ result

grad1, val1 = gradient(g1, xtest)
grad2, val2 = gradient(g2, xtest)

gradsum = randn(size(xtest))
valsum = gradient!(gradsum, g, xtest)
@test gradsum ≈ grad1 + grad2
