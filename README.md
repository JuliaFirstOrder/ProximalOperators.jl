# ProximalOperators.jl

[![Build Status](https://travis-ci.org/kul-forbes/ProximalOperators.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/ProximalOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/00rk6ip0y0t2wj8t?svg=true)](https://ci.appveyor.com/project/lostella/prox-jl)
[![Coverage Status](https://coveralls.io/repos/github/kul-forbes/ProximalOperators.jl/badge.svg?branch=master)](https://coveralls.io/github/kul-forbes/ProximalOperators.jl?branch=master)

Proximal operators for nonsmooth optimization in Julia.
This package can be used to easily implement proximal algorithms for convex and nonconvex optimization problems such as ADMM, the alternating direction method of multipliers.

## Installation

To install the stable version, use the following in the Julia command line

```julia
Pkg.add("ProximalOperators")
```

If instead you are interested in the (possibly unstable) development version, use

```julia
Pkg.clone("https://github.com/kul-forbes/ProximalOperators.jl.git")
```

Remember to `Pkg.update()` to keep the package up to date.

## Usage

With `using ProximalOperators` the package exports the `prox` and `prox!` methods to evaluate the proximal mapping of several functions.
Such functions can be instantiated using *constructors*: the available constructors are listed [here](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/FUNCTIONS.md).
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
julia> fx = prox!(f, x, 0.5) # in-place equivalent to x, fx = prox(f, x, 0.5)
julia> fy = prox!(f, x, y, 0.5) # in-place equivalent to y, fy = prox(f, x, 0.5)
```

## Contributing

If you wish to contribute to the repository,
see the [to do list](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/TODO.md).

## References

1. Neal Parikh and Stephen Boyd, [*Proximal Algorithms*](http://dx.doi.org/10.1561/2400000003).
Foundations and Trends in Optimization 1, 3 (2014), 127-239.

## Credits

ProximalOperators.jl is developed by
[Lorenzo Stella](https://lostella.github.io)
and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/)
at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/).
