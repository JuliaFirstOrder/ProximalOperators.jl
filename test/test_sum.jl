using Random

Random.seed!(123)

# smooth case

f1 = SqrNormL2()
f2 = Translate(SqrNormL2(2.5), randn(10))
f = Sum(f1, f2)

predicates_test(f)

@test is_quadratic(f) == true
@test is_strongly_convex(f) == true
@test is_set_indicator(f) == false

xtest = randn(10)

result = f1(xtest) + f2(xtest)
@test f(xtest) ≈ result

grad1, val1 = gradient_test(f1, xtest)
grad2, val2 = gradient_test(f2, xtest)

gradsum, valsum = gradient_test(f, xtest)
@test gradsum ≈ grad1 + grad2

# nonsmooth case

g1 = NormL2()
g2 = Translate(SqrNormL2(2.5), randn(10))
g = Sum(g1, g2)

predicates_test(g)

@test is_smooth(g) == false
@test is_strongly_convex(g) == true
@test is_set_indicator(g) == false

xtest = randn(10)

result = g1(xtest) + g2(xtest)
@test g(xtest) ≈ result

grad1, val1 = gradient_test(g1, xtest)
grad2, val2 = gradient_test(g2, xtest)

gradsum, valsum = gradient_test(g, xtest)
@test gradsum ≈ grad1 + grad2
