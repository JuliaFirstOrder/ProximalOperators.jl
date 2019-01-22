using Documenter, ProximalOperators

makedocs(
    modules = [ProximalOperators],
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
    repo   = "github.com/kul-forbes/ProximalOperators.jl.git",
)
