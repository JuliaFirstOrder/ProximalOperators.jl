srand(123)

f1 = SqrNormL2()
f2 = Translate(SqrNormL2(2.5), randn(10))
f = Sum(f1, f2)
xtest = randn(10)

result = f1(xtest) + f2(xtest)
@test f(xtest) ≈ result

grad1, val1 = gradient(f1, xtest)
grad2, val2 = gradient(f2, xtest)

gradsum = ones(xtest)
valsum = gradient!(gradsum, f, xtest)
@test gradsum ≈ grad1 + grad2
