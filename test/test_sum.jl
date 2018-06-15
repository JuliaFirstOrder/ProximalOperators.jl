srand(123)

f1 = NormL2()
f2 = NormL2()
f = Sum(f1, f2)
xtest = randn(10)

result = f1(xtest) + f2(xtest)
@test f(xtest) ≈ result

grad1, val1 = gradient(f1, xtest)
grad2, val2 = gradient(f2, xtest)

gradsum, valsum = gradient(f, xtest)
@test gradsum ≈ grad1 + grad2
