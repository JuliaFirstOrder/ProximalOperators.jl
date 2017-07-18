## TODO

### General

* Tests should be reorganized.

### Functions

* Indicator of L1 norm ball. (TODO: faster projection [Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191) or [Condat 2016](http://link.springer.com/article/10.1007/s10107-015-0946-6).)
* Sum of k-largest components. (TODO: fix, see [sumLargest.jl](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/src/functions/sumLargest.jl).)
* Ky-Fan (k)-norms. (Should be easy once one has the previous.)
* Iterative `prox!` for `LeastSquares`: a new type `LeastSquaresIterative`.
* Iterative `prox!` for `IndAffine`: a new type `IndAffineIterative`.
* Iterative `prox!` for generic smooth functions: this could be a type `IterativeProx` which wraps a given smooth function `f` and uses the best suited iterative method to compute `prox!`, for example based on the functions properties (is `f` quadratic? Yes: CG, No: nonlinear CG). Should `LeastSquares` and `Quadratic` be included in this more general case case?

### Calculus rules

* Epi-composition. (See [epicompose.jl](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/src/calculus/epicompose.jl).)
