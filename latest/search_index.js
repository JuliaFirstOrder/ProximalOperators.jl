var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ProximalOperators.jl-1",
    "page": "Home",
    "title": "ProximalOperators.jl",
    "category": "section",
    "text": "ProximalOperators is a Julia package that implements first-order primitives for a variety of functions, which are commonly used for implementing optimization algorithms in several application areas, e.g., statistical learning, image and signal processing, optimal control.Please refer to the GitHub repository to browse the source code, report issues and submit pull requests."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install the package, use the following in the Julia command linePkg.add(\"ProximalOperators\")To load the package simply typeusing ProximalOperatorsRemember to do Pkg.update() from time to time, to keep the package up to date."
},

{
    "location": "index.html#Quick-introduction-1",
    "page": "Home",
    "title": "Quick introduction",
    "category": "section",
    "text": "For a function f and a stepsize gamma  0, the proximal operator (or proximal mapping) is given bymathrmprox_gamma f(x) = argmin_z left f(z) + tfrac12gammaz-x^2 rightand can be efficiently computed for many functions f used in applications.ProximalOperators allows to pick function f from a library of commonly used functions, and to modify and combine them using calculus rules to obtain new ones. The proximal mapping of f is then provided through the prox and prox! methods, as described here.For example, one can create the L1-norm as follows.julia> using ProximalOperators\n\njulia> f = NormL1(3.5)\ndescription : weighted L1 norm\ndomain      : AbstractArray{Real}, AbstractArray{Complex}\nexpression  : x ↦ λ||x||_1\nparameters  : λ = 3.5Functions created this way are, of course, callable.julia> x = [1.0, 2.0, 3.0, 4.0, 5.0]; # some point\n\njulia> f(x)\n52.5Method prox evaluates the proximal operator associated with a function, given a point and (optionally) a positive stepsize parameter, returning the proximal point y and the value of the function at y:julia> y, fy = prox(f, x, 0.5) # last argument is 1.0 if absent\n([0.0, 0.25, 1.25, 2.25, 3.25], 24.5)Method prox! evaluates the proximal operator in place, and only returns the function value at the proximal point (in this case y must be preallocated and have the same shape/size as x):julia> y = similar(x); # allocate y\n\njulia> fy = prox!(y, f, x, 0.5) # in-place equivalent to y, fy = prox(f, x, 0.5)\n24.5"
},

{
    "location": "index.html#Bibliographic-references-1",
    "page": "Home",
    "title": "Bibliographic references",
    "category": "section",
    "text": "N. Parikh and S. Boyd (2014), Proximal Algorithms, Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239.\nS. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein (2011), Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers, Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122."
},

{
    "location": "index.html#Credits-1",
    "page": "Home",
    "title": "Credits",
    "category": "section",
    "text": "ProximalOperators.jl is developed by Lorenzo Stella and Niccolò Antonello at KU Leuven, ESAT/Stadius, and Mattias Fält at Lunds Universitet, Department of Automatic Control."
},

{
    "location": "functions.html#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "Here we list the available functions, grouped by category. Each function is documented with its exact definition and the necessary parameters for construction. The proximal mapping (and gradient, when defined) of such functions is computed by calling the prox and prox! methods (and gradient, gradient!, when defined). These functions can be modified and/or combined together to make new ones, by means of calculus rules."
},

{
    "location": "functions.html#ProximalOperators.IndAffine",
    "page": "Functions",
    "title": "ProximalOperators.IndAffine",
    "category": "Type",
    "text": "Indicator of an affine subspace\n\nIndAffine(A, b)\n\nIf A is a matrix (dense or sparse) and b is a vector, returns the indicator function of the set\n\nS = x  Ax = b\n\nIf A is a vector and b is a scalar, returns the indicator function of the set\n\nS = x  langle A x rangle = b\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBallLinf",
    "page": "Functions",
    "title": "ProximalOperators.IndBallLinf",
    "category": "Function",
    "text": "Indicator of a L_ norm ball\n\nIndBallLinf(r=1.0)\n\nReturns the indicator function of the set\n\nS =  x  max (x_i) leq r \n\nParameter r must be positive.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBallL0",
    "page": "Functions",
    "title": "ProximalOperators.IndBallL0",
    "category": "Type",
    "text": "Indicator of a L_0 pseudo-norm ball\n\nIndBallL0(r=1)\n\nReturns the indicator function of the set\n\nS =  x  mathrmnnz(x) leq r \n\nParameter r must be a positive integer.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBallL1",
    "page": "Functions",
    "title": "ProximalOperators.IndBallL1",
    "category": "Type",
    "text": "Indicator of a L_1 norm ball\n\nIndBallL1(r=1.0)\n\nReturns the indicator function of the set\n\nS = left x  _i x_i leq r right\n\nParameter r must be positive.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBallL2",
    "page": "Functions",
    "title": "ProximalOperators.IndBallL2",
    "category": "Type",
    "text": "Indicator of a Euclidean ball\n\nIndBallL2(r=1.0)\n\nReturns the indicator function of the set\n\nS =  x  x leq r \n\nwhere cdot is the L_2 (Euclidean) norm. Parameter r must be positive.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBallRank",
    "page": "Functions",
    "title": "ProximalOperators.IndBallRank",
    "category": "Type",
    "text": "Indicator of rank ball\n\nIndBallRank(r=1)\n\nReturns the indicator function of the set of matrices of rank at most r:\n\nS =  X  mathrmrank(X) leq r \n\nParameter r must be a positive integer.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBinary",
    "page": "Functions",
    "title": "ProximalOperators.IndBinary",
    "category": "Type",
    "text": "Indicator of the product of binary sets\n\nIndBinary(low, up)\n\nReturns the indicator function of the set\n\nS =  x  x_i = low_i textor x_i = up_i \n\nParameters low and up can be either scalars or arrays of the same dimension as the space.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndBox",
    "page": "Functions",
    "title": "ProximalOperators.IndBox",
    "category": "Type",
    "text": "Indicator of a box\n\nIndBox(low, up)\n\nReturns the indicator function of the set\n\nS =  x  low leq x leq up \n\nParameters low and up can be either scalars or arrays of the same dimension as the space: they must satisfy low <= up, and are allowed to take values -Inf and +Inf to indicate unbounded coordinates.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndHalfspace",
    "page": "Functions",
    "title": "ProximalOperators.IndHalfspace",
    "category": "Type",
    "text": "Indicator of a halfspace\n\nIndHalfspace(a, b)\n\nFor an array a and a scalar b, returns the indicator of set\n\nS = x  langle ax rangle leq b \n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndPoint",
    "page": "Functions",
    "title": "ProximalOperators.IndPoint",
    "category": "Type",
    "text": "Indicator of a singleton\n\nIndPoint(p=0.0)\n\nReturns the indicator of the set\n\nC = p \n\nParameter p can be a scalar, in which case the unique element of S has uniform coefficients.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndSimplex",
    "page": "Functions",
    "title": "ProximalOperators.IndSimplex",
    "category": "Type",
    "text": "Indicator of a simplex\n\nIndSimplex(a=1.0)\n\nReturns the indicator of the set\n\nS = left x  x geq 0 _i x_i = a right\n\nBy default a=1.0, therefore S is the probability simplex.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndSphereL2",
    "page": "Functions",
    "title": "ProximalOperators.IndSphereL2",
    "category": "Type",
    "text": "Indicator of a Euclidean sphere\n\nIndSphereL2(r=1.0)\n\nReturns the indicator function of the set\n\nS =  x  x = r \n\nwhere cdot is the L_2 (Euclidean) norm. Parameter r must be positive.\n\n\n\n"
},

{
    "location": "functions.html#Indicators-of-sets-1",
    "page": "Functions",
    "title": "Indicators of sets",
    "category": "section",
    "text": "When function f is the indicator function of a set S, that isf(x) = _S(x) =\nbegincases\n0  textif x in S \n+  textotherwise\nendcasesthen mathrmprox_f = _S is the projection onto S. Therefore ProximalOperators includes in particular projections onto commonly used sets, which are here listed.IndAffine\nIndBallLinf   \nIndBallL0     \nIndBallL1     \nIndBallL2     \nIndBallRank   \nIndBinary\nIndBox       \nIndHalfspace  \nIndPoint              \nIndSimplex    \nIndSphereL2          "
},

{
    "location": "functions.html#ProximalOperators.IndExpPrimal",
    "page": "Functions",
    "title": "ProximalOperators.IndExpPrimal",
    "category": "Type",
    "text": "Indicator of the (primal) exponential cone\n\nIndExpPrimal()\n\nReturns the indicator function of the primal exponential cone, that is\n\nC = mathrmcl  (rst)  s  0 se^rs leq t  subset mathbbR^3\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndExpDual",
    "page": "Functions",
    "title": "ProximalOperators.IndExpDual",
    "category": "Function",
    "text": "Indicator of the (dual) exponential cone\n\nIndExpDual()\n\nReturns the indicator function of the dual exponential cone, that is\n\nC = mathrmcl  (uvw)  u  0 -ue^vu leq we  subset mathbbR^3\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndFree",
    "page": "Functions",
    "title": "ProximalOperators.IndFree",
    "category": "Type",
    "text": "Indicator of the free cone\n\nIndFree()\n\nReturns the indicator function of the whole space, or \"free cone\", i.e., a function which is identically zero.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndNonnegative",
    "page": "Functions",
    "title": "ProximalOperators.IndNonnegative",
    "category": "Type",
    "text": "Indicator of the nonnegative orthant\n\nIndNonnegative()\n\nReturns the indicator of the set\n\nC =  x  x geq 0 \n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndNonpositive",
    "page": "Functions",
    "title": "ProximalOperators.IndNonpositive",
    "category": "Type",
    "text": "Indicator of the nonpositive orthant\n\nIndNonpositive()\n\nReturns the indicator of the set\n\nC =  x  x leq 0 \n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndPSD",
    "page": "Functions",
    "title": "ProximalOperators.IndPSD",
    "category": "Type",
    "text": "Indicator of the set of positive semi-definite cone\n\nIndPSD()\n\nReturns the indicator of the set\n\nC =  X  X succeq 0 \n\nThe argument to the function can be either a Symmetric or Hermitian object, or an object of type AbstractVector{Float64} holding a symmetric matrix in (lower triangular) packed storage.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndSOC",
    "page": "Functions",
    "title": "ProximalOperators.IndSOC",
    "category": "Type",
    "text": "Indicator of the second-order cone\n\nIndSOC()\n\nReturns the indicator of the second-order cone (also known as ice-cream cone or Lorentz cone), that is\n\nC = left (t x)  x leq t right\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndRotatedSOC",
    "page": "Functions",
    "title": "ProximalOperators.IndRotatedSOC",
    "category": "Type",
    "text": "Indicator of the rotated second-order cone\n\nIndRotatedSOC()\n\nReturns the indicator of the rotated second-order cone (also known as ice-cream cone or Lorentz cone), that is\n\nC = left (p q x)  x^2 leq 2cdot pq p geq 0 q geq 0 right\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.IndZero",
    "page": "Functions",
    "title": "ProximalOperators.IndZero",
    "category": "Type",
    "text": "Indicator of the zero cone\n\nIndZero()\n\nReturns the indicator function of the set containing the origin, the \"zero cone\".\n\n\n\n"
},

{
    "location": "functions.html#Indicators-of-convex-cones-1",
    "page": "Functions",
    "title": "Indicators of convex cones",
    "category": "section",
    "text": "An important class of sets in optimization is that of convex cones. These are used in particular for formulating cone programming problems, a family of problems which includes linear programs (LP), quadratic programs (QP), quadratically constrained quadratic programs (QCQP) and semidefinite programs (SDP).IndExpPrimal\nIndExpDual\nIndFree\nIndNonnegative\nIndNonpositive\nIndPSD\nIndSOC\nIndRotatedSOC\nIndZero"
},

{
    "location": "functions.html#ProximalOperators.ElasticNet",
    "page": "Functions",
    "title": "ProximalOperators.ElasticNet",
    "category": "Type",
    "text": "Elastic-net regularization\n\nElasticNet(μ=1.0, λ=1.0)\n\nReturns the function\n\nf(x) = x_1 + (2)x^2\n\nfor nonnegative parameters μ and λ.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NormL0",
    "page": "Functions",
    "title": "ProximalOperators.NormL0",
    "category": "Type",
    "text": "L_0 pseudo-norm\n\nNormL0(λ=1.0)\n\nReturns the function\n\nf(x) = cdotmathrmnnz(x)\n\nfor a nonnegative parameter λ.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NormL1",
    "page": "Functions",
    "title": "ProximalOperators.NormL1",
    "category": "Type",
    "text": "L_1 norm\n\nNormL1(λ=1.0)\n\nWith a nonnegative scalar parameter λ, returns the function\n\nf(x) = cdot_ix_i\n\nWith a nonnegative array parameter λ, returns the function\n\nf(x) = _i _ix_i\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NormL2",
    "page": "Functions",
    "title": "ProximalOperators.NormL2",
    "category": "Type",
    "text": "L_2 norm\n\nNormL2(λ=1.0)\n\nWith a nonnegative scalar parameter λ, returns the function\n\nf(x) = cdotsqrtx_1^2 +  + x_n^2\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NormL21",
    "page": "Functions",
    "title": "ProximalOperators.NormL21",
    "category": "Type",
    "text": "Sum-of-L_2 norms\n\nNormL21(λ=1.0, dim=1)\n\nReturns the function\n\nf(X) = _ix_i\n\nfor a nonnegative λ, where x_i is the i-th column of X if dim == 1, and the i-th row of X if dim == 2. In words, it is the sum of the Euclidean norms of the columns or rows.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NormLinf",
    "page": "Functions",
    "title": "ProximalOperators.NormLinf",
    "category": "Function",
    "text": "L_ norm\n\nNormLinf(λ=1.0)\n\nReturns the function\n\nf(x) = maxx_1  x_n\n\nfor a nonnegative parameter λ.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.NuclearNorm",
    "page": "Functions",
    "title": "ProximalOperators.NuclearNorm",
    "category": "Type",
    "text": "Nuclear norm\n\nNuclearNorm(λ=1.0)\n\nReturns the function\n\nf(X) = X_* =  _i _i(X)\n\nwhere λ is a positive parameter and _i(X) is i-th singular value of matrix X.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.SqrNormL2",
    "page": "Functions",
    "title": "ProximalOperators.SqrNormL2",
    "category": "Type",
    "text": "Squared Euclidean norm (weighted)\n\nSqrNormL2(λ=1.0)\n\nWith a nonnegative scalar λ, returns the function\n\nf(x) = tfrac2x^2\n\nWith a nonnegative array λ, returns the function\n\nf(x) = tfrac12_i _i x_i^2\n\n\n\n"
},

{
    "location": "functions.html#Norms-and-regularization-functions-1",
    "page": "Functions",
    "title": "Norms and regularization functions",
    "category": "section",
    "text": "ElasticNet\nNormL0\nNormL1\nNormL2\nNormL21\nNormLinf\nNuclearNorm\nSqrNormL2"
},

{
    "location": "functions.html#ProximalOperators.HingeLoss",
    "page": "Functions",
    "title": "ProximalOperators.HingeLoss",
    "category": "Function",
    "text": "Hinge loss\n\nHingeLoss(b, μ=1.0)\n\nReturns the function\n\nf(x) = _i max0 1 - b_i  x_i\n\nwhere b is an array and μ is a positive parameter.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.HuberLoss",
    "page": "Functions",
    "title": "ProximalOperators.HuberLoss",
    "category": "Type",
    "text": "Huber loss\n\nHuberLoss(ρ=1.0, μ=1.0)\n\nReturns the function\n\nf(x) = begincases\n  tfrac2x^2  textif x   \n  (x - tfrac2)  textotherwise\nendcases\n\nwhere ρ and μ are positive parameters.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.LeastSquares",
    "page": "Functions",
    "title": "ProximalOperators.LeastSquares",
    "category": "Type",
    "text": "Least squares penalty\n\nLeastSquares(A, b, λ=1.0)\n\nFor a matrix A, a vector b and a scalar λ, returns the function\n\nf(x) = tfraclambda2Ax - b^2\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.Linear",
    "page": "Functions",
    "title": "ProximalOperators.Linear",
    "category": "Type",
    "text": "Linear function\n\nLinear(c)\n\nReturns the function\n\nf(x) = langle c x rangle\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.LogBarrier",
    "page": "Functions",
    "title": "ProximalOperators.LogBarrier",
    "category": "Type",
    "text": "Logarithmic barrier\n\nLogBarrier(a=1.0, b=0.0, μ=1.0)\n\nReturns the function\n\nf(x) = -_ilog(ax_i+b)\n\nfor a nonnegative parameter μ.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.Maximum",
    "page": "Functions",
    "title": "ProximalOperators.Maximum",
    "category": "Function",
    "text": "Maximum coefficient\n\nMaximum(λ=1.0)\n\nFor a nonnegative parameter λ ⩾ 0, returns the function\n\nf(x) = lambda cdot max x_i  i = 1ldots n \n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.Quadratic",
    "page": "Functions",
    "title": "ProximalOperators.Quadratic",
    "category": "Type",
    "text": "Quadratic function\n\nQuadratic(Q, q)\n\nFor a matrix Q (dense or sparse, symmetric and positive definite) and a vector q, returns the function\n\nf(x) = tfrac12langle Qx xrangle + langle q x rangle\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.QuadraticIterative",
    "page": "Functions",
    "title": "ProximalOperators.QuadraticIterative",
    "category": "Type",
    "text": "Quadratic function (iterative evaluation of prox)\n\nQuadraticIterative(Q, q)\n\nFor a matrix Q (dense or sparse, symmetric and positive definite) and a vector q, returns the function\n\nf(x) = tfrac12langle Qx xrangle + langle q x rangle\n\nDifferently from Quadratic, in this case the prox operation is evaluated (inexactly) using an iterative method.\n\n\n\n"
},

{
    "location": "functions.html#ProximalOperators.SumPositive",
    "page": "Functions",
    "title": "ProximalOperators.SumPositive",
    "category": "Type",
    "text": "Sum of the positive coefficients\n\nSumPositive()\n\nReturns the function\n\nf(x) = _i max0 x_i\n\n\n\n"
},

{
    "location": "functions.html#Penalties-and-other-functions-1",
    "page": "Functions",
    "title": "Penalties and other functions",
    "category": "section",
    "text": "HingeLoss   \nHuberLoss   \nLeastSquares\nLinear\nLogBarrier\nMaximum\nQuadratic\nQuadraticIterative\nSumPositive"
},

{
    "location": "calculus.html#",
    "page": "Calculus rules",
    "title": "Calculus rules",
    "category": "page",
    "text": ""
},

{
    "location": "calculus.html#Calculus-rules-1",
    "page": "Calculus rules",
    "title": "Calculus rules",
    "category": "section",
    "text": "The calculus rules described in the following allow to modify and combine functions, to obtain new ones with efficiently computable proximal mapping."
},

{
    "location": "calculus.html#ProximalOperators.Conjugate",
    "page": "Calculus rules",
    "title": "ProximalOperators.Conjugate",
    "category": "Type",
    "text": "Convex conjugate\n\nConjugate(f)\n\nReturns the convex conjugate (also known as Fenchel conjugate, or Fenchel-Legendre transform) of function f, that is\n\nf^*(x) = sup_y  langle y x rangle - f(y) \n\n\n\n"
},

{
    "location": "calculus.html#Duality-1",
    "page": "Calculus rules",
    "title": "Duality",
    "category": "section",
    "text": "Conjugate"
},

{
    "location": "calculus.html#ProximalOperators.DistL2",
    "page": "Calculus rules",
    "title": "ProximalOperators.DistL2",
    "category": "Type",
    "text": "Distance from a convex set\n\nDistL2(ind_S)\n\nGiven ind_S the indicator function of a convex set S, and an optional positive parameter λ, returns the (weighted) Euclidean distance from S, that is function\n\ng(x) = mathrmdist_S(x) = min  y - x  y in S \n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.SqrDistL2",
    "page": "Calculus rules",
    "title": "ProximalOperators.SqrDistL2",
    "category": "Type",
    "text": "Squared distance from a convex set\n\nSqrDistL2(ind_S, λ=1.0)\n\nGiven ind_S the indicator function of a convex set S, and an optional positive parameter λ, returns the (weighted) squared Euclidean distance from S, that is function\n\ng(x) = tfrac2mathrmdist_S^2(x) = min left tfrac2y - x^2  y in S right\n\n\n\n"
},

{
    "location": "calculus.html#Distances-from-convex-sets-1",
    "page": "Calculus rules",
    "title": "Distances from convex sets",
    "category": "section",
    "text": "When the indicator of a convex set is constructed (see Indicators of sets) the (squared) distance from the set can be constructed using the following:DistL2\nSqrDistL2"
},

{
    "location": "calculus.html#ProximalOperators.SeparableSum",
    "page": "Calculus rules",
    "title": "ProximalOperators.SeparableSum",
    "category": "Type",
    "text": "Separable sum of functions\n\nSeparableSum(f₁,…,fₖ)\n\nGiven functions f₁ to fₖ, returns their separable sum, that is\n\ng(x_1x_k) = _i=1^k f_i(x_i)\n\nThe object g constructed in this way can be evaluated at Tuples of length k. Likewise, the prox and prox! methods for g operate with (input and output) Tuples of length k.\n\nExample:\n\nf = SeparableSum(NormL1(), NuclearNorm()); # separable sum of two functions\nx = randn(10); # some random vector\nY = randn(20, 30); # some random matrix\nf_xY = f((x, Y)); # evaluates f at (x, Y)\n(u, V), f_uV = prox(f, (x, Y), 1.3); # computes prox at (x, Y)\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.SlicedSeparableSum",
    "page": "Calculus rules",
    "title": "ProximalOperators.SlicedSeparableSum",
    "category": "Type",
    "text": "Sliced separable sum of functions\n\nSlicedSeparableSum((f₁,…,fₖ), (J₁,…,Jₖ))\n\nReturns the function\n\ng(x) = _i=1^k f_i(x_J_i)\n\nSlicedSeparableSum(f, (J₁,…,Jₖ))\n\nAnalogous to the previous one, but applies the same function f to all slices of the variable x:\n\ng(x) = _i=1^k f(x_J_i)\n\n\n\n"
},

{
    "location": "calculus.html#Functions-combination-1",
    "page": "Calculus rules",
    "title": "Functions combination",
    "category": "section",
    "text": "The following means of combination are important in that they allow to represent a very common situation: defining the sum of multiple functions, each applied to an independent block of variables. The following two constructors, SeparableSum and SlicedSeparableSum, allow to do this in two (complementary) ways.SeparableSum\nSlicedSeparableSum"
},

{
    "location": "calculus.html#ProximalOperators.MoreauEnvelope",
    "page": "Calculus rules",
    "title": "ProximalOperators.MoreauEnvelope",
    "category": "Type",
    "text": "Moreau envelope\n\nMoreauEnvelope(f, γ=1.0)\n\nReturns the Moreau envelope (also known as Moreau-Yosida regularization) of function f with parameter γ (positive), that is\n\nf^(x) = min_z left f(z) + tfrac12z-x^2 right\n\nIf f is convex, then f^ is a smooth, convex, lower approximation to f, having the same minima as the original function.\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.Regularize",
    "page": "Calculus rules",
    "title": "ProximalOperators.Regularize",
    "category": "Type",
    "text": "Regularize\n\nRegularize(f, ρ=1.0, a=0.0)\n\nGiven function f, and optional parameters ρ (positive) and a, returns\n\ng(x) = f(x) + tfrac2x-a\n\nParameter a can be either an array or a scalar, in which case it is subtracted component-wise from x in the above expression.\n\n\n\n"
},

{
    "location": "calculus.html#Functions-regularization-1",
    "page": "Calculus rules",
    "title": "Functions regularization",
    "category": "section",
    "text": "MoreauEnvelope\nRegularize"
},

{
    "location": "calculus.html#ProximalOperators.Postcompose",
    "page": "Calculus rules",
    "title": "ProximalOperators.Postcompose",
    "category": "Type",
    "text": "Postcomposition with an affine transformation\n\nPostcompose(f, a=1.0, b=0.0)\n\nReturns the function\n\ng(x) = acdot f(x) + b\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.Precompose",
    "page": "Calculus rules",
    "title": "ProximalOperators.Precompose",
    "category": "Type",
    "text": "Precomposition with linear mapping/translation\n\nPrecompose(f, L, μ, b)\n\nReturns the function\n\ng(x) = f(Lx + b)\n\nwhere f is a convex function and L is a linear mapping: this must satisfy LL^* = I for   0. Furthermore, either f is separable or parameter μ is a scalar, for the prox of g to be computable.\n\nParameter L defines L through the A_mul_B! and Ac_mul_B! methods. Therefore L can be an AbstractMatrix for example, but not necessarily.\n\nIn this case, prox and prox! are computed according to Prop. 24.14 in Bauschke, Combettes \"Convex Analisys and Monotone Operator Theory in Hilbert Spaces\", 2nd edition, 2016. The same result is Prop. 23.32 in the 1st edition of the same book.\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.PrecomposeDiagonal",
    "page": "Calculus rules",
    "title": "ProximalOperators.PrecomposeDiagonal",
    "category": "Type",
    "text": "Precomposition with diagonal scaling/translation\n\nPrecomposeDiagonal(f, a, b)\n\nReturns the function\n\ng(x) = f(mathrmdiag(a)x + b)\n\nwhere f is a convex function. Furthermore, f must be separable, or a must be a scalar, for the prox of g to be computable. Parametes a and b can be arrays of multiple dimensions, according to the shape/size of the input x that will be provided to the function: the way the above expression for g should be thought of, is g(x) = f(a.*x + b).\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.Tilt",
    "page": "Calculus rules",
    "title": "ProximalOperators.Tilt",
    "category": "Type",
    "text": "Linear tilting\n\nTilt(f, a, b=0.0)\n\nGiven function f, an array a and a constant b (optional), returns function\n\ng(x) = f(x) + langle a x rangle + b\n\n\n\n"
},

{
    "location": "calculus.html#ProximalOperators.Translate",
    "page": "Calculus rules",
    "title": "ProximalOperators.Translate",
    "category": "Type",
    "text": "Translation\n\nTranslate(f, b)\n\nReturns the translated function\n\ng(x) = f(x + b)\n\n\n\n"
},

{
    "location": "calculus.html#Pre-and-post-transformations-1",
    "page": "Calculus rules",
    "title": "Pre- and post-transformations",
    "category": "section",
    "text": "Postcompose\nPrecompose\nPrecomposeDiagonal\nTilt\nTranslate"
},

{
    "location": "operators.html#",
    "page": "Prox and gradient",
    "title": "Prox and gradient",
    "category": "page",
    "text": ""
},

{
    "location": "operators.html#ProximalOperators.prox",
    "page": "Prox and gradient",
    "title": "ProximalOperators.prox",
    "category": "Function",
    "text": "Proximal mapping\n\nprox(f, x, γ=1.0)\n\nComputes\n\ny = mathrmprox_gamma f(x) = argmin_z left f(z) + tfrac12gammaz-x^2 right\n\nThe resulting point y is returned as first output, and f(y) as second output.\n\n\n\n"
},

{
    "location": "operators.html#ProximalOperators.prox!",
    "page": "Prox and gradient",
    "title": "ProximalOperators.prox!",
    "category": "Function",
    "text": "Proximal mapping (in-place)\n\nprox!(y, f, x, γ=1.0)\n\nComputes\n\ny = mathrmprox_gamma f(x) = argmin_z left f(z) + tfrac12gammaz-x^2 right\n\nThe resulting point y is written to the (pre-allocated) array y, which must have the same shape/size as x, and the value the proximal point of x with respect to function f(y) is returned.\n\n\n\n"
},

{
    "location": "operators.html#Base.LinAlg.gradient",
    "page": "Prox and gradient",
    "title": "Base.LinAlg.gradient",
    "category": "Function",
    "text": "Gradient mapping\n\ngradient(f, x)\n\nFor a differentiable function f, returns nabla f(x) as first output, and f(x) as second output.\n\n\n\n"
},

{
    "location": "operators.html#ProximalOperators.gradient!",
    "page": "Prox and gradient",
    "title": "ProximalOperators.gradient!",
    "category": "Function",
    "text": "Gradient mapping (in-place)\n\ngradient!(y, f, x)\n\nFor a differentiable function f, writes nabla f(x) to y, which must be pre-allocated and have the same shape/size as x, and returns f(x) as output.\n\n\n\n"
},

{
    "location": "operators.html#Prox-and-gradient-1",
    "page": "Prox and gradient",
    "title": "Prox and gradient",
    "category": "section",
    "text": "The following methods allow to evaluate the proximal mapping (and gradient, when defined) of mathematical functions, which are constructed according to what described in Functions and Calculus rules.prox\nprox!\ngradient\ngradient!"
},

{
    "location": "operators.html#Complex-and-matrix-variables-1",
    "page": "Prox and gradient",
    "title": "Complex and matrix variables",
    "category": "section",
    "text": "The proximal mapping is usually discussed in the case of functions over mathbbR^n. However, by adapting the inner product langlecdotcdotrangle and associated norm cdot adopted in its definition, one can extend the concept to functions over more general spaces. When functions of unidimensional arrays (vectors) are concerned, the standard Euclidean product and norm are used in defining prox (therefore prox!, but also gradient and gradient!). This are the inner product and norm which are computed by dot and norm in Julia.When bidimensional, tridimensional (matrices and tensors) and higher dimensional arrays are concerned, then the definitions of proximal mapping and gradient are naturally extended by considering the appropriate inner product. For k-dimensional arrays, of size n_1 times n_2 times ldots times n_k, we consider the inner productlangle A B rangle = sum_i_1ldotsi_k A_i_1ldotsi_k cdot B_i_1ldotsi_kwhich reduces to the usual Euclidean product in case of unidimensional arrays, and to the trace product langle A B rangle = mathrmtr(A^top B) in the case of matrices (bidimensional arrays). This inner product, and the associated norm, are the ones computed by vecdot and vecnorm in Julia."
},

{
    "location": "operators.html#Multiple-variable-blocks-1",
    "page": "Prox and gradient",
    "title": "Multiple variable blocks",
    "category": "section",
    "text": "By combining functions together through SeparableSum, the resulting function will have multiple inputs, i.e., it will be defined over the Cartesian product of the domains of the individual functions. To represent elements (points) of such product space, here we use Julia's Tuple objects.Example. Suppose that the following function needs to be represented:f(x Y) = x_1 + Y_*that is, the sum of the L_1 norm of some vector x and the nuclear norm (the sum of the singular values) of some matrix Y. This is accomplished as follows:using ProximalOperators\nf = SeparableSum(NormL1(), NuclearNorm());Now, function f is defined over pairs of appropriate Array objects. Likewise, the prox method will take pairs of Arrays as inputs, and return pairs of Arrays as output:x = randn(10); # some random vector\nY = randn(20, 30); # some random matrix\nf_xY = f((x, Y)); # evaluates f at (x, Y)\n(u, V), f_uV = prox(f, (x, Y), 1.3); # computes prox at (x, Y)The same holds for the separable sum of more than two functions, in which case \"pairs\" are to be replaced with Tuples of the appropriate length."
},

{
    "location": "demos.html#",
    "page": "Demos",
    "title": "Demos",
    "category": "page",
    "text": ""
},

{
    "location": "demos.html#Demos-1",
    "page": "Demos",
    "title": "Demos",
    "category": "section",
    "text": "The demos folder contains examples on how to use the functions of ProximalOperators to implement optimization algorithms. Warning: Make sure that the version of ProximalOperators that you have installed is up-to-date with the demo script you are trying to run, as the package features may change over time and the master branch be ahead of what you have installed."
},

]}
