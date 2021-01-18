using LinearAlgebra
using Random

Random.seed!(123)

# CASE 1

x = randn(10)
y0 = randn(10)
y = copy(y0)

prox_col = (NormL1(0.1),IndBallL0(1))
ind_col = ((1:5,),(6:10,))

f = SlicedSeparableSum(prox_col,ind_col)
y, fy = prox(f,x,1.)
yn,fyn = ProximalOperators.prox_naive(f,x,1.)
y1,fy1 = prox(prox_col[1],x[ind_col[1]...],1.)
y2,fy2 = prox(prox_col[2],x[ind_col[2]...],1.)

@test abs(f(y)-fy)<1e-11
@test abs(fyn-fy)<1e-11
@test norm(yn-y)<1e-11
@test abs((fy1+fy2)-fy)<1e-11
@test norm(y-[y1;y2])<1e-11

# CASE 2

X1,X2 = randn(10,10),randn(10,10)
X = [X1; X2]

f = SlicedSeparableSum((NormL1(1.), NormL21(0.1)), ((1:10,:),(11:20,:)))

y,fy = prox(f,X,1.)
yn,fyn = ProximalOperators.prox_naive(f,X,1.)
y1,fy1 = prox(NormL1(1.),X1,1.)
y2,fy2 = prox(NormL21(0.1),X2,1.)

@test abs(f(y)-fy)<1e-11
@test abs(fyn-fy)<1e-11
@test norm(yn-y)<1e-11
@test abs((fy1+fy2)-fy)<1e-11
@test norm(y-[y1; y2])<1e-11

# CASE 3

x1, x2, x3 = randn(10), randn(10), randn(10)
X = [x1 x2 x3]

f = NormL2()
F = SlicedSeparableSum(f, ((:,1),(:,2),(:,3)))

y,Fy = prox(F,X,1.)
yn,Fyn = ProximalOperators.prox_naive(F,X,1.)
y1,fy1 = prox(f,x1,1.)
y2,fy2 = prox(f,x2,1.)
y3,fy3 = prox(f,x3,1.)

@test abs(F(y)-Fy)<1e-11
@test abs(Fyn-Fy)<1e-11
@test norm(yn-y)<1e-11
@test abs((fy1+fy2+fy3)-Fy)<1e-11
@test norm(y-[y1 y2 y3])<1e-11

# CASE 4

x = randn(10)
y0 = randn(10)
y = copy(y0)

prox_col = (NormL1(0.1),IndBallL0(1))
ind_col = ((collect(1:5),),(collect(6:10),))

f = SlicedSeparableSum(prox_col,ind_col)
y, fy = prox(f,x,1.)
yn,fyn = ProximalOperators.prox_naive(f,x,1.)
y1,fy1 = prox(prox_col[1],x[ind_col[1]...],1.)
y2,fy2 = prox(prox_col[2],x[ind_col[2]...],1.)

@test abs(f(y)-fy)<1e-11
@test abs(fyn-fy)<1e-11
@test norm(yn-y)<1e-11
@test abs((fy1+fy2)-fy)<1e-11
@test norm(y-[y1;y2])<1e-11

# Test with Quadratic (iterative)

Q = randn(5,10)
Q = Q'*Q
q = randn(10)
x = randn(10)
xx = vcat(x, x)
f = Quadratic(Q, q, iterative=true)
g = SlicedSeparableSum((f, f), ((1:10,), (11:20,)))
y, fy = prox(f, x)
yy, fyy = prox(g, xx)

@test yy ≈ vcat(y, y)
@test fyy ≈ 2*fy
