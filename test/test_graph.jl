## Test sparse
#  workspace()
# using ProximalOperators
#
# using Base.Test

srand(0)

m, n = (10, 50)

A = sprand(m, n, 0.3)
Adf = full(A)
B = [A -speye(m)]
Bd = full(B)



fiag = ProximalOperators.IndGraph(A)
fiaf = ProximalOperators.IndAffine(B, zeros(m))
fiagdf = ProximalOperators.IndGraph(Adf)

c = randn(n) * 10 + 10
d = A * c

@test fiag(c, d) == 0.0
@test fiagdf(c, d) == 0.0

y = zeros(d)
x = zeros(c)

xy = [x ; y]
cd = [c ; d]

x_a = @view xy[1:n]
y_a = @view xy[n+1:end]

## IndAffine QR
prox!(xy, fiaf, cd)
@test fiaf(xy) == 0.0

## SPARSE
# test call
prox!(x, y, fiag, c, d)
@test fiag(x, y) == 0.0
# test vs IndAffine
@test norm(x_a - x, Inf) < TOL_ASSERT
@test norm(y_a - y, Inf) < TOL_ASSERT


## FAT
# test call
prox!(x, y, fiagdf, c, d)
@test fiagdf(x, y) <= TOL_ASSERT
# test versus IndAffine qr variant
@test norm(x_a - x, Inf) < TOL_ASSERT
@test norm(y_a - y, Inf) < TOL_ASSERT
