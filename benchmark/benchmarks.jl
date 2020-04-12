using ProximalOperators
using BenchmarkTools
using LinearAlgebra

const SUITE = BenchmarkGroup()

k = "IndPSD"
SUITE[k] = BenchmarkGroup(["IndPSD"])
for T in [Float32, Float64, Complex{Float32}, Complex{Float64}]
    for n in [3, 10, 20, 50]
        SUITE[k][T, n] = @benchmarkable prox!(Y, f, X) setup=begin
            f = IndPSD()
            W = if $T <: Real
                Symmetric
            else
                Hermitian
            end
            X = begin
                A = randn($T, $n, $n)
                W((A + A')/2)
            end
            Y = similar(X)
        end
    end
end
