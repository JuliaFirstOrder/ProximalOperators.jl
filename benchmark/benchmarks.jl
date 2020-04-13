using ProximalOperators
using BenchmarkTools
using LinearAlgebra
using Random

const SUITE = BenchmarkGroup()

k = "IndPSD"
SUITE[k] = BenchmarkGroup(["IndPSD"])
for T in [Float64, Complex{Float64}]
    for n in [10, 20, 50]
        SUITE[k][T, n] = @benchmarkable prox!(Y, f, X) setup=begin
            f = IndPSD()
            W = if $T <: Real
                Symmetric
            else
                Hermitian
            end
            Random.seed!(0)
            A = randn($T, $n, $n)
            X = W((A + A')/2)
            Y = similar(X)
        end
    end
end

k = "IndBox"
SUITE[k] = BenchmarkGroup(["IndBox"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        low = -ones($T, 10000)
        upp = +ones($T, 10000)
        f = IndBox(low, upp)
        x = [-2*ones($T, 3000); zeros($T, 4000); ones($T, 3000)]
        y = similar(x)
    end
end

k = "IndNonnegative"
SUITE[k] = BenchmarkGroup(["IndNonnegative"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndNonnegative()
        x = [-2*ones($T, 3000); zeros($T, 4000); ones($T, 3000)]
        y = similar(x)
    end
end

k = "NormL2"
SUITE[k] = BenchmarkGroup(["NormL2"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = NormL2()
        x = ones($T, 10000)
        y = similar(x)
    end
end
