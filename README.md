# ProximalOperators.jl

[![Build status](https://github.com/JuliaFirstOrder/ProximalOperators.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaFirstOrder/ProximalOperators.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaFirstOrder/ProximalOperators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaFirstOrder/ProximalOperators.jl)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ProximalOperators-jl/Lobby)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4020558.svg)](https://doi.org/10.5281/zenodo.4020558)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafirstorder.github.io/ProximalOperators.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliafirstorder.github.io/ProximalOperators.jl/latest)


Proximal operators for nonsmooth optimization in Julia.
This package can be used to easily implement proximal algorithms for convex and nonconvex optimization problems such as ADMM, the alternating direction method of multipliers.

See the [documentation](https://juliafirstorder.github.io/ProximalOperators.jl/latest) on how to use the package.

## Installation

To install the package, hit `]` from the Julia command line to enter the package manager, then

```julia
pkg> add ProximalOperators
```

## Usage

With `using ProximalOperators` the package exports the `prox` and `prox!` methods to evaluate the proximal mapping of several functions.

A list of available function constructors is in the [documentation](https://juliafirstorder.github.io/ProximalOperators.jl/latest).

For example, you can create the L1-norm as follows.

```julia
julia> f = NormL1(3.5)
description : weighted L1 norm
type        : Array{Complex} → Real
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

Functions created this way are, of course, callable.

```julia
julia> x = randn(10) # some random point
julia> f(x)
32.40700818735099
```

**`prox`** evaluates the proximal operator associated with a function,
given a point and (optionally) a positive stepsize parameter,
returning the proximal point `y` and the value of the function at `y`:

```julia
julia> y, fy = prox(f, x, 0.5) # last argument is 1.0 if absent
```

**`prox!`** evaluates the proximal operator *in place*,
and only returns the function value at the proximal point:

```julia
julia> fy = prox!(y, f, x, 0.5) # in-place equivalent to y, fy = prox(f, x, 0.5)
```

## Related packages

* [FirstOrderSolvers.jl](https://github.com/mfalt/FirstOrderSolvers.jl)
* [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
* [StructuredOptimization.jl](https://github.com/JuliaFirstOrder/StructuredOptimization.jl)

## References

1. N. Parikh and S. Boyd (2014), [*Proximal Algorithms*](http://dx.doi.org/10.1561/2400000003),
Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239.

2. S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein (2011), [*Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*](http://dx.doi.org/10.1561/2200000016), Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122.

## Credits

ProximalOperators.jl is developed by
[Lorenzo Stella](https://lostella.github.io)
and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/)
at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/),
and [Mattias Fält](http://www.control.lth.se/Staff/MattiasFalt.html) at [Lunds Universitet, Department of Automatic Control](http://www.control.lth.se/).
