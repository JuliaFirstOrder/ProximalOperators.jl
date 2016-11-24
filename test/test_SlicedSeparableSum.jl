using Base.Test
using ProximalOperators

srand(123)
x = randn(100)
y0 = randn(100)
y = copy(y0)

prox_col = [NormL1(0.1),IndBallL0(1)]
ind_col = [1:50,51:100]
M1 = ones(Bool,size(x))
M1[51:end] = false
M2 = M1.==false

f = SlicedSeparableSum(prox_col,ind_col)
#test second constructor and mask
ff = SlicedSeparableSum(prox_col[1]=>M1,prox_col[2]=>M2)
y,fy = prox(f,x,1.)
yy,fyy = prox(ff,x,1.)

@test norm(fyy-fy)<1e-11
@test norm(y-yy)<1e-11

y1,fy1 = prox(prox_col[1],x[ind_col[1]],1.)
y2,fy2 = prox(prox_col[2],x[ind_col[2]],1.)

@test norm((fy1+fy2)-fy)<1e-11
@test norm(y-[y1;y2])<1e-11

X1,X2 = randn(10,10),randn(10,10)
X = [X1 X2]

f = SlicedSeparableSum([NormL1(1.),NormL21(0.1)],[1:10,11:20],2 )
#test second constructor
ff = SlicedSeparableSum(NormL1(1.)=> 1:10, NormL21(0.1)=> 11:20; dim =2 )

y,fy = prox(f,X,1.)
yy,fyy = prox(ff,X,1.)

@test norm(fyy-fy)<1e-11
@test norm(y-yy)<1e-11

y1,fy1 = prox(NormL1(1.),X1,1.)
y2,fy2 = prox(NormL21(0.1),X2,1.)

@test norm((fy1+fy2)-fy)<1e-11
@test norm(y-[y1 y2])<1e-11

show(ff)
show(SlicedSeparableSum(repmat([NormL1(1.)],6),repmat([1:10],6),2 ))
