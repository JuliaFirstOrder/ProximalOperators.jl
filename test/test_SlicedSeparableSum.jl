using Base.Test
using ProximalOperators

srand(123)
x = randn(10)
y0 = randn(10)
y = copy(y0)

prox_col = [NormL1(0.1),IndBallL0(1)]
ind_col = [(1:5,),(6:10,)]

f = SlicedSeparableSum(prox_col,ind_col)
y, fy = prox(f,x,1.)

y1,fy1 = prox(prox_col[1],x[ind_col[1]...],1.)
y2,fy2 = prox(prox_col[2],x[ind_col[2]...],1.)

@test norm((fy1+fy2)-fy)<1e-11
@test norm(y-[y1;y2])<1e-11

X1,X2 = randn(10,10),randn(10,10)
X = [X1; X2]

f = SlicedSeparableSum([NormL1(1.), NormL21(0.1)], [(1:10,:),(11:20,:)])

y,fy = prox(f,X,1.)

y1,fy1 = prox(NormL1(1.),X1,1.)
y2,fy2 = prox(NormL21(0.1),X2,1.)

@test norm((fy1+fy2)-fy)<1e-11
@test norm(y-[y1; y2])<1e-11

@test_throws ErrorException SlicedSeparableSum([NormL1(1.), NormL21(0.1)], [(1.0,:),(11:20,:)])
