using Documenter, ProximalOperators, ProximalCore

makedocs(
    modules = [ProximalOperators, ProximalCore],
    sitename = "ProximalOperators.jl",
    pages = [
        "Home" => "index.md",
        "Functions" => "functions.md",
        "Calculus rules" => "calculus.md",
        "Prox and gradient" => "operators.md",
        "Demos" => "demos.md"
    ],
)

deploydocs(
    repo   = "github.com/JuliaFirstOrder/ProximalOperators.jl.git",
)
