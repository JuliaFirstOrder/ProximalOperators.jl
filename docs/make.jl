using Documenter, ProximalOperators

makedocs(
  modules = [ProximalOperators]
)

deploydocs(
  deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-bootstrap"),
  repo   = "github.com/kul-forbes/ProximalOperators.jl.git",
  julia  = "0.6",
  osname = "linux"
)
