## TODO

* Update docstrings.
* Add more tests, especially with `Array{Complex}` variables.
* Implement calculus rules.
* Implement iterative routines to evaluate general proximal mappings.

## Functions to add

* Indicator of L1 norm ball. ([Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191))
* Least squares penalty.
* Generic quadratic function.
* Ky-Fan (k)-norms.
* Indicator of exponential cone. (Numerically)

## Calculus rules

* Moreau identity (conjugation)
* Precomposition

## Questions

* In `prox!`, is it safe to use `@inbounds`? Because we take the iterator for `x`
and not for `y`, in principle one could provide `y` of the wrong dimension.
