using ProximalOperators, Random
# Number of variables
n = 1000
# Number of halfspaces
mi = 50 # Inequalities with C
me = 50 # Equalities with A

Random.seed!(1)
# One point in polytope
x0 = randn(n)

# Create polytope containing x0
# Inequality
C = Matrix{Float64}(undef, mi, n)
d = randn(mi)

# Make sure x0 is in polytope by setting sign of inequality, random C part
for i = 1:mi
    v = randn(n)
    b = randn()
    if v'x0  <= b
        C[i,:] .= v
    else
        C[i,:] .= -v
    end
    d[i] = b
end

# Create equality
A = randn(me, n)
b = A*x0

l = [b;fill(-Inf, mi)]
u = [b;d]
AC = [A;C]

# Precompile
polyOSQP  = IndPolyhedral(l, AC, u; solver=:osqp)
polyQPDAS = IndPolyhedral(l, AC, u; solver=:qpdas)
x = randn(n)
y = similar(x0)
prox!(y, polyOSQP, x)
prox!(y, polyQPDAS, x)
# Run tests
println("Setup OSQP")
@time polyOSQP  = IndPolyhedral(l, AC, u; solver=:osqp)
println("Setup QPDAS")
@time polyQPDAS = IndPolyhedral(l, AC, u; solver=:qpdas)

Random.seed!(2)
x = randn(n)
y = similar(x0)
println("First prox OSQP:")
@time prox!(y, polyOSQP, x)

Random.seed!(2)
x = randn(n)
println("First prox QPDAS:")
@time prox!(y, polyQPDAS, x)

Random.seed!(3)
N = 100
xs = randn(n, N)
x = xs[:,1]

println("100 prox! OSQP:")
@time for i = 1:100
    # Project from here
    x .= xs[:,i]
    prox!(y, polyOSQP, x)
end

Random.seed!(3)
N = 100
xs = randn(n, N)
x = xs[:,1]

println("100 prox! QPDAS:")
@time for i = 1:100
    # Project from here
    x .= xs[:,i]
    prox!(y, polyQPDAS, x)
end
