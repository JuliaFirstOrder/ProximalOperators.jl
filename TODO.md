## TODO

### General

* Fix docstrings (in particular: types of constructors arguments).
* Add more tests, especially with `Array{Complex}` variables.
* Implement iterative routines to evaluate general proximal mappings.
* For separable functions it makes sense to have `gamma::Array`.

### Functions to add

* Least squares penalty.
* Generic quadratic function.
* Indicator of L1 norm ball. (TODO: faster projection [Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191) or [Condat 2016](http://link.springer.com/article/10.1007/s10107-015-0946-6).)
* Sum of k-largest components. ([Parikh, Boyd 2014](http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf).)
* Ky-Fan (k)-norms.
* Max function.
* Log barrier.

### Calculus rules

* Pre-composition with uniform scaling. (To be improved.)
* Moreau identity. (To be improved.)
* Separable sum. (Need to find a way to represent blocks of variables.)
* (Block-)Diagonal pre-scaling and translation. (For (block-)separable functions. Requires `gamma::Array`, see above.)
* Pre-composition by generic affine mapping.
