# ProximalOperators.jl

ProximalOperators is a [Julia](https://julialang.org) package that implements first-order primitives for a variety of functions, which are commonly used for implementing optimization algorithms in several application areas, *e.g.*, statistical learning, image and signal processing, optimal control.

Please refer to the [GitHub repository](https://github.com/kul-forbes/ProximalOperators.jl) to browse the source code, report issues and submit pull requests.

## Installation

To install the package, hit `]` from the Julia command line to enter the package manager, then

```julia
pkg> add ProximalOperators
```

To load the package simply type

```julia
using ProximalOperators
```

Remember to do `Pkg.update()` from time to time, to keep the package up to date.

## Quick introduction

For a function ``f`` and a stepsize ``\gamma > 0``, the *proximal operator* (or *proximal mapping*) is given by
```math
\mathrm{prox}_{\gamma f}(x) = \arg\min_z \left\{ f(z) + \tfrac{1}{2\gamma}\|z-x\|^2 \right\}
```
and can be efficiently computed for many functions ``f`` used in applications.

ProximalOperators allows to pick function ``f`` from a [library of commonly used functions](functions.md), and to modify and combine them using [calculus rules](calculus.md) to obtain new ones. The proximal mapping of ``f`` is then provided through the [`prox`](@ref) and [`prox!`](@ref) methods, as described [here](operators.md).

For example, one can create the L1-norm as follows.

```jldoctest quickex1
julia> using ProximalOperators

julia> f = NormL1(3.5)
description : weighted L1 norm
domain      : AbstractArray{Real}, AbstractArray{Complex}
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

Functions created this way are, of course, callable.

```jldoctest quickex1
julia> x = [1.0, 2.0, 3.0, 4.0, 5.0]; # some point

julia> f(x)
52.5
```

Method [`prox`](@ref) evaluates the proximal operator associated with a function,
given a point and (optionally) a positive stepsize parameter,
returning the proximal point `y` and the value of the function at `y`:

```jldoctest quickex1
julia> y, fy = prox(f, x, 0.5) # last argument is 1.0 if absent
([0.0, 0.25, 1.25, 2.25, 3.25], 24.5)
```

Method [`prox!`](@ref) evaluates the proximal operator *in place*,
and only returns the function value at the proximal point (in this case `y` must be preallocated and have the same shape/size as `x`):

```jldoctest quickex1
julia> y = similar(x); # allocate y

julia> fy = prox!(y, f, x, 0.5) # in-place equivalent to y, fy = prox(f, x, 0.5)
24.5
```

## Bibliographic references

1. N. Parikh and S. Boyd (2014), [*Proximal Algorithms*](http://dx.doi.org/10.1561/2400000003), Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239.

2. S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein (2011), [*Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*](http://dx.doi.org/10.1561/2200000016), Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122.

## Credits

ProximalOperators.jl is developed by
[Lorenzo Stella](https://lostella.github.io)
and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/)
at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/),
and [Mattias Fält](http://www.control.lth.se/Staff/MattiasFalt.html) at [Lunds Universitet, Department of Automatic Control](http://www.control.lth.se/).
