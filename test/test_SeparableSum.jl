using Base.Test
using ProximalOperators

srand(123)
x = randn(100)
y0 = randn(100)
y = copy(y0)

prox_col = [NormL1(0.1),NormL2(0.01)]
ind_col = [1:50,51:100]

f = SeparableSum(prox_col,ind_col)
y,fy = prox(f,x,1.)

y1,fy1 = prox(prox_col[1],x[ind_col[1]],1.)
y2,fy2 = prox(prox_col[2],x[ind_col[2]],1.)

@test norm((fy1+fy2)-fy)<1e-11
@test norm(y-[y1;y2])<1e-11


X = randn(100,100)
prox_col = (NormL21(0.1),NormL2(0.01))
ind_col = (1:50,51:100)

