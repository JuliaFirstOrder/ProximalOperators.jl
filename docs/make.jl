using Documenter, ProximalOperators

makedocs(
  modules = [ProximalOperators],
  format = :html,
  sitename = "ProximalOperators.jl",
  authors = "Lorenzo Stella and contributors",
  pages = Any[
  "Home" => "index.md",
  "Functions" => "functions.md",
  "Calculus rules" => "calculus.md",
  "Prox and gradient" => "operators.md",
  "Demos" => "demos.md"
  ],
)

deploydocs(
  repo   = "github.com/kul-forbes/ProximalOperators.jl.git",
  julia  = "1.0",
  osname = "linux",
  target = "build",
  deps = nothing,
  make = nothing,
)
