using Documenter, ProximalOperators

makedocs(modules = [ProximalOperators],
         format = :html,
         sitename = "ProximalOperators.jl",
         authors = "Lorenzo Stella",
         pages = Any[
         "Home" => "index.md",
         "Functions" => "functions.md",
         "Calculus rules" => "calculus.md",
         "Demos" => "demos.md"
         ]
         )
