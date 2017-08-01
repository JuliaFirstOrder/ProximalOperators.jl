## Test sparse
# workspace()
# using ProximalOperators
# using Base.Test

rng = MersenneTwister(1234)

m, n = (10, 50)

A = sprand( rng, Float64, m, n, 0.3)
Adf = full(A)
Ads = Adf';
B = [A -speye(m)]
Bd = full(B)

fiag = ProximalOperators.IndGraph(A)
fiaf = ProximalOperators.IndAffine(B, zeros(m))
fiagdf = ProximalOperators.IndGraph(Adf)
fiagds = ProximalOperators.IndGraph(Ads)

c = randn(rng, Float64, n) * 10 + 10
d = A * c

v = randn(rng, Float64, m) * 10 + 10
w = Ads * v

@test fiag((c, d)) == 0.0
@test fiagdf(c, d) == 0.0
@test fiagds(v, w) == 0.0

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
( xn, yn, res) = ProximalOperators.prox_naive(fiag, c,d)
@test fiag(xn, yn) == 0.0
# test naive vs. prox!
@test norm(xn - x, Inf) < 1e-10
@test norm(yn - y, Inf) < 1e-10
# test vs IndAffine
@test norm(x_a - x, Inf) < 1e-10
@test norm(y_a - y, Inf) < 1e-10
# test tuple signature
prox!((x,y), fiag, (c,d))
@test fiag((x,y)) == 0.0

## FAT
# test call
prox!(x, y, fiagdf, c, d)
@test fiagdf(x, y) < 1e-10
( xn, yn, res) = ProximalOperators.prox_naive(fiagdf, c,d)
@test fiagdf(xn, yn) == 0.0
# test naive vs. prox!
@test norm(xn - x, Inf) < 1e-10
@test norm(yn - y, Inf) < 1e-10
# test versus IndAffine qr variant
@test norm(x_a - x, Inf) < 1e-10
@test norm(y_a - y, Inf) < 1e-10
# test tuple signature
prox!((x,y), fiagdf, (c,d))
@test fiagdf((x,y)) == 0.0

## SKINNY
# test call
prox!(y, x, fiagds, v, w)
@test fiagds(y, x) < 1e-10
( yn, xn, res) = ProximalOperators.prox_naive(fiagds, v,w)
@test fiagds(yn, xn) == 0.0
# test naive vs. prox!
@test norm(xn - x, Inf) < 1e-10
@test norm(yn - y, Inf) < 1e-10
# test tuple signature
prox!((y,x), fiagds, (v,w))
@test fiagds((y,x)) == 0.0


## MISC
# test calls to other functions
for foo in [fiag, fiagdf, fiagds]
  @test ProximalOperators.fun_name(foo) != ""
  @test ProximalOperators.fun_dom(foo) != ""
  @test ProximalOperators.fun_expr(foo) != ""
  @test ProximalOperators.fun_params(foo) != ""
  @test ProximalOperators.is_convex(foo) == true
  @test ProximalOperators.is_set(foo) == true
  @test ProximalOperators.is_cone(foo) == true
end
