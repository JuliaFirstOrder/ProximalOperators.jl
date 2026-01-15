using Documenter, ProximalOperators, ProximalCore

DocMeta.setdocmeta!(ProximalCore, :DocTestSetup, :(import ProximalCore); recursive=true)

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
    checkdocs=:none,
)

deploydocs(
    repo   = "github.com/JuliaFirstOrder/ProximalOperators.jl.git",
)
