## TODO

### General

* Add more tests, especially with `Array{Complex}` variables.
* Implement iterative routines to evaluate general proximal mappings.
* For separable functions it makes sense to have `gamma::Array`.

### Functions to add

* Generic quadratic function.
* Indicator of L1 norm ball. (TODO: faster projection [Duchi et al. 2008](http://dl.acm.org/citation.cfm?id=1390191) or [Condat 2016](http://link.springer.com/article/10.1007/s10107-015-0946-6).)
* Sum of k-largest components. (TODO: fix, see `SumLargest.jl`.)
* Ky-Fan (k)-norms. (Easy once one has the previous)

### Calculus rules

* Separable sum. (Need to find a way to represent blocks of variables.)
* (Block-)Diagonal pre-scaling and translation. (For (block-)separable functions. Requires `gamma::Array`, see above.)
* Pre-composition by generic affine mapping.
