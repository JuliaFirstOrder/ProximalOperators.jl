
using ProximalOperators
using PyPlot
using DataFrames

srand(123)
t = [0.4;0]
f = Translate(NormL1(),-t)

r = 0.15
g = IndBallL2(r)

theta = linspace(0,2*pi,50)
xi,yi = r*cos.(theta).+t[1]/2,r*sin.(theta).+t[1]
x = [xi yi]' 

yf = zeros(x)
yg = zeros(x)

for i =1:size(yf,2)
	yf[:,i] = prox(f,x[:,i],0.4)[1]
	yg[:,i] = prox(g,x[:,i],1)[1]
end

figure()
i = 0
for  d = linspace(0,r,5) 
	i += 1
	xi = t[1].+[0; -d; 0; d; 0; -d]
	yi = t[2].+[d; 0; -d; 0; d;  0]
	plot(xi,yi, "r")
	writetable("l1_$i.cvs", DataFrame(x = round.(xi,4) , y = round.(yi,4) ))
end
theta = linspace(0,2*pi+pi/10,100)
xi,yi = r*cos.(theta),r*sin.(theta)
plot(xi,yi,"g")
writetable("idl2.cvs", DataFrame(x = round.(xi,4) , y = round.(yi,4) ))
for i = 1:size(yg,2)
	plot([x[1,i];yg[1,i]],[x[2,i];yg[2,i]],"g")
	writetable("prox1_$i.cvs", DataFrame(x = round.([x[1,i];yg[1,i]],4) , y = round.([x[2,i];yg[2,i]],4) ))
end
for i = 1:size(yf,2)
	plot([x[1,i];yf[1,i]],[x[2,i];yf[2,i]],"r")
	writetable("prox2_$i.cvs", DataFrame(x = round.([x[1,i];yf[1,i]],4) , y = round.([x[2,i];yf[2,i]],4) ))
end

plot(x[1,:],x[2,:],"*m")
writetable("datapoints.cvs", DataFrame(x = round.(x[1,:],4) , y = round.(x[2,:],4) ))

