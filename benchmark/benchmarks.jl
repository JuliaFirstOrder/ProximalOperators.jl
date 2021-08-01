using ProximalOperators
using BenchmarkTools
using LinearAlgebra
using SparseArrays
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

k = "LeastSquares"
SUITE[k] = BenchmarkGroup(["LeastSquares"])
for (T, s, sparse, iterative) in Iterators.product(
    [Float64, ComplexF64],
    [(5, 11), (11, 5)],
    [false, true],
    [false, true],
)
    SUITE[k][(T, s, sparse, iterative)] = @benchmarkable prox!(y, f, x) setup=begin
        A = if $sparse sparse(ones($T, $s)) else ones($T, $s) end
        b = ones($T, $(s[1]))
        f = LeastSquares(A, b, iterative=$iterative)
        x = ones($T, $(s[2]))
        y = similar(x)
    end
end

k = "IndExpPrimal"
SUITE[k] = BenchmarkGroup(["IndExpPrimal"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndExpPrimal()
        x = [0.537667139546100, 1.833885014595086, -2.258846861003648]
        y = similar(x)
    end
end

k = "IndSimplex"
SUITE[k] = BenchmarkGroup(["IndSimplex"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndSimplex()
        x = collect($T, -0.5:0.001:2.0)
        y = similar(x)
    end
end

k = "IndBallL1"
SUITE[k] = BenchmarkGroup(["IndBallL1"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndBallL1()
        x = collect($T, -2.0:0.001:0.5)
        y = similar(x)
    end
end
