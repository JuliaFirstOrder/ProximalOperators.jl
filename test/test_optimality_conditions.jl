# test whether prox satisfies necessary conditions for optimality

using LinearAlgebra
using SparseArrays
using ProximalOperators
using Test

check_optimality(f::LeastSquares, x, gamma, y) = norm(y + gamma * f.lambda * (f.A' * (f.A * y - f.b)) - x) <= 1e-10
check_optimality(f::HuberLoss, x, gamma, y) = isapprox((x - y) / gamma, (norm(y) <= f.rho ? f.mu * y : f.rho * f.mu * y / norm(y)))
check_optimality(f::SqrHingeLoss, x, gamma, y) = isapprox((x - y) / gamma, -2 .* f.mu .* f.y .* max.(0, 1 .- f.y .* y))
check_optimality(::IndSimplex, x, _, y) = begin
    w = x - y
    tau = dot(w, y) / sum(y)
    all(w .<= tau + 10 * eps(eltype(x)))
end
check_optimality(f::IndBallL1, x, gamma, y) = begin
    if norm(x, 1) <= f.r
        return all(y .== x)
    end
    sign_is_correct = (sign.(y) .== 0) .| (sign.(x) .== sign.(y))
    return all(sign_is_correct) && check_optimality(IndSimplex(f.r), abs.(x), gamma, abs.(y))
end

check_optimality(f::TotalVariation1D, x, gamma, y) = begin
    N = length(x)
    # compute dual solution
    u = zeros(N+1)
    u[1] = 0
    for k in 2:N+1
        u[k] = x[k-1]-y[k-1]+u[k-1]
    end

    # check whether all duals in interval
    c1 = all(abs.(u) .<= gamma*f.lambda + 1e-10)
    # check whether last equals 0 (first by construction)
    c2 = isapprox(u[end], 0, atol=1e-12)
    # check whether equal +- gamma*lambda
    h = sign.(y[1:end-1] - y[2:end])
    c3 = all(isapprox.( u[2:end-1] .* abs.(h) , h *f.lambda*gamma))
    return c1 && c2 && c3
end

test_cases = [
    Dict(
        "f" => LeastSquares(randn(20, 10), randn(20)),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => LeastSquares(randn(15, 40), randn(15), rand()),
        "x" => randn(40),
        "gamma" => rand(),
    ),

    Dict(
        "f" => LeastSquares(rand(Complex{Float64}, 15, 40), rand(Complex{Float64}, 15), rand()),
        "x" => rand(Complex{Float64}, 40),
        "gamma" => rand(),
    ),

    Dict(
        "f" => LeastSquares(sprandn(100,1000,0.05), randn(100), rand()),
        "x" => randn(1000),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndSimplex(),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndSimplex(2),
        "x" => [1.5, 0.0, 0.5, 0.0, 0.0],
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndSimplex(5.0),
        "x" => [0.5, 0.5, 1.0, 1.0, 2.0],
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndSimplex(rand()),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndBallL1(),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndBallL1(1.7),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndBallL1(5.0),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => IndBallL1(rand()),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => HuberLoss(),
        "x" => randn(10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => HuberLoss(rand()),
        "x" => randn(8, 10),
        "gamma" => rand(),
    ),

    Dict(
        "f" => HuberLoss(rand(), rand()),
        "x" => randn(20),
        "gamma" => rand(),
    ),

    Dict(
        "f" => HuberLoss(rand(), rand()),
        "x" => rand(Complex{Float64}, 12, 15),
        "gamma" => rand(),
    ),

    Dict(
        "f" => SqrHingeLoss(randn(5)),
        "x" => randn(5),
        "gamma" => 0.1 + rand(),
    ),

    Dict(
        "f" => SqrHingeLoss(randn(5), 0.1+rand()),
        "x" => randn(5),
        "gamma" => 0.1 + rand(),
    ),

    Dict(
        "f" => SqrHingeLoss(randn(3, 5), 0.1+rand()),
        "x" => randn(3, 5),
        "gamma" => 0.1 + rand(),
    ),
]

@testset "Optimality conditions" begin
    @testset "$(typeof(d["f"]))" for d in test_cases
        f, x, gamma = d["f"], d["x"], d["gamma"]
        y, fy = prox(f, x, gamma)
        @test check_optimality(f, x, gamma, y)
    end
end
