## TODO

* Update docstrings.
* Add more tests, especially with `Array{Complex}` variables.
* Implement calculus rules.
* Implement iterative routines to evaluate general proximal mappings.

## Functions to add

* Least squares penalty.
* Generic quadratic function.
* Indicator of L1 norm ball. ([Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191))
* Sum of k-largest components. ([Parikh, Boyd 2014](http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf))
* Ky-Fan (k)-norms.
* Max function.
* Log barrier.
* Indicator of exponential cone. (Numerically)

## Calculus rules

* Moreau identity (conjugation)
* Precomposition

## Questions

* In `prox!`, is it safe to use `@inbounds`? Because we take the iterator for `x`
and not for `y`, in principle one could provide `y` of the wrong dimension.
