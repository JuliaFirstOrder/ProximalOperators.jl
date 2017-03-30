## TODO

### General

* Tests, tests, tests. Especially with `Array{Complex}` variables and `gamma::Array`.
* Organization of the tests could also be re-thought: it would be nice to have only a batch of tests of the type "do this, then verify these properties".
* For separable functions it makes sense to have `gamma::Array`. (Work in progress: see [sumPositive.jl](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/src/functions/sumPositive.jl).)
* Make a list of functions with their mathematical expression, properties (e.g.: separable/non-separable).
* Documentation, documentation, documentation.

### Functions

* Generic quadratic function.
* Indicator of L1 norm ball. (TODO: faster projection [Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191) or [Condat 2016](http://link.springer.com/article/10.1007/s10107-015-0946-6).)
* Sum of k-largest components. (TODO: fix, see [sumLargest.jl](https://github.com/kul-forbes/ProximalOperators.jl/blob/master/src/functions/sumLargest.jl).)
* Ky-Fan (k)-norms. (Should be easy once one has the previous)
* Numerical evaluation of `prox` for generic smooth function

### Calculus rules

* Pre-composition with a generic affine mapping. (Work in progress.)
