############
# dspev!
############

if isdefined(Prox, :dspev!) && isdefined(Prox, :dspevV!)

println("testing dspev!")

a = [1.0,2.0,3.0,5.0,6.0,9.0]
W_ref = [0.0,0.6992647456322766,14.300735254367698]
Z_ref = [0.9486832980505137	0.17781910596911388	-0.26149639682478454;
  0.0	-0.8269242138935418	-0.5623133863572413;
  -0.3162277660168381	0.5334573179073402	-0.7844891904743537]
A_ref = [1.0 2.0 3.0; 2.0 5.0 6.0; 3.0 6.0 9.0]

a_copy = copy(a); W1, Z1 = Prox.dspev!('V','L',a_copy)
a_copy = copy(a); W2, Z2 = Prox.dspevV!('L',a_copy)

A1 = Z1*diagm(W1)*Z1'
A2 = Z2*diagm(W2)*Z2'

@test all((W1-W_ref)./(1+abs(W_ref)) .<= 1e-8)
@test all((A1-A_ref)./(1+abs(A_ref)) .<= 1e-8)
@test all((W2-W_ref)./(1+abs(W_ref)) .<= 1e-8)
@test all((A2-A_ref)./(1+abs(A_ref)) .<= 1e-8)

end
