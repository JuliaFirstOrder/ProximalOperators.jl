# IndPolyhedral: QPDAS implementation

import QPDAS

"""
**Indicator of a polyhedral set**

    IndPolyhedralQPDAS(A, C, b, d)

```math
S = \\{x : \\langle A, x \\rangle = b\\ ∧ \\langle C, x \\rangle \\le b\\}.
```
"""
struct IndPolyhedralQPDAS{R<:Real, MT<:AbstractMatrix{R}, VT<:AbstractVector{R}, QP<:QPDAS.QuadraticProgram} <: IndPolyhedral
    A::MT
    b::VT
    C::MT
    d::VT
    z::VT
    qp::QP
    first_prox::Ref{Bool}
    function IndPolyhedralQPDAS{R}(A::MT, b::VT, C::MT, d::VT) where {R<:Real, MT<:AbstractMatrix{R}, VT<:AbstractVector{R}, QP<:QPDAS.QuadraticProgram}
        @assert size(A,1) == size(b,1)
        qp = QPDAS.QuadraticProgram(A, b, C, d, smartstart=false)
        new{R, MT, VT, typeof(qp)}(A, b, C, d, zeros(R,size(A,2)), qp, Ref(true))
    end
end

# properties

is_prox_accurate(::IndPolyhedralQPDAS) = true

# constructors

function IndPolyhedralQPDAS(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
) where R
    if !all(l .<= u)
        error("function is improper (are some bounds inverted?)")
    end
    eqinds = (l .== u) .& .!isnothing.(l)
    Aeq = A[eqinds,:]
    beq = l[eqinds]

    _islower(l::T) where T =
        l != typemin(T) && !isnan(l) && !isnothing(l)
    _isupper(u::T) where T =
        u != typemax(T) && !isnan(u) && !isnothing(u)

    lower = _islower.(l) .& (.!eqinds)
    upper = _isupper.(u) .& (.!eqinds)

    lower_only = lower .& (.! upper)
    upper_only = upper .& (.! lower)
    upper_and_lower = upper .& lower


    Cieq = [-A[lower_only, :];
             A[upper_only, :];
            -A[upper_and_lower, :];
             A[upper_and_lower, :] ]
    dieq = [-l[lower_only];
             u[upper_only];
            -l[upper_and_lower];
             u[upper_and_lower] ]

    IndPolyhedralQPDAS{R}(Aeq, beq, Cieq, dieq)
end

IndPolyhedralQPDAS(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R},
    xmin::AbstractVector{R}, xmax::AbstractVector{R}
) where R =
    IndPolyhedralQPDAS([l; xmin], [A; I], [u; xmax])

IndPolyhedralQPDAS(
    l::AbstractVector{R}, A::AbstractMatrix{R}, args...
) where R =
    IndPolyhedralQPDAS(
        l, A, R(Inf).*ones(R, size(A, 1)), args...
    )

IndPolyhedralQPDAS(
    A::AbstractMatrix{R}, u::AbstractVector{R}, args...
) where R =
    IndPolyhedralQPDAS(
        R(-Inf).*ones(R, size(A, 1)), A, u, args...
    )

# function evaluation

function (f::IndPolyhedralQPDAS{R})(x::AbstractVector{R}) where R
    Ax = f.A * x
    Cx = f.C * x
    return all(Ax .<= f.b .& Cx .<= f.d) ? R(0) : Inf
end

# prox

function prox!(y::AbstractVector{R}, f::IndPolyhedralQPDAS{R}, x::AbstractVector{R}, gamma::R=R(1)) where R
    # Linear term in qp is -x
    f.z .= .- x
    # Update the problem
    QPDAS.update!(f.qp, z=f.z)

    if f.first_prox[]
        # This sets the initial active set based on z, should only be run once
        QPDAS.run_smartstart(f.qp.boxQP)
        f.first_prox[] = false
    end
    sol, val = QPDAS.solve!(f.qp)
    y .= sol
    return R(0)
end

# naive prox

# we want to compute the projection p of a point x
#
# primal problem is: minimize_p (1/2)||p-x||^2 + g(Ap)
# where g is the indicator of the box [l, u]
#
# dual problem is: minimize_y (1/2)||-A'y||^2 - x'A'y + g*(y)
# can solve with (fast) dual proximal gradient method

function prox_naive(f::IndPolyhedralQPDAS{R}, x::AbstractVector{R}, gamma::R=R(1)) where R
    # Rewrite to l ≤ Ax ≤ u
    A = [f.A; f.C]
    l = [f.b; fill(R(-Inf), length(f.d))]
    u = [f.b; f.d]
    y = zeros(R, size(A, 1)) # dual vector
    y1 = y
    g = IndBox(l, u)
    gstar = Conjugate(g)
    gstar_y = R(0)
    stepsize = R(1)/opnorm(Matrix(A*A'))
    for it = 1:1e6
        w = y + (it-1)/(it+2)*(y - y1)
        y1 = y
        z = w - stepsize * (A * (A'*w - x))
        y, = prox(gstar, z, stepsize)
        if norm(y-w)/(1+norm(w)) <= 1e-12 break end
    end
    p = -A'*y + x
    return p, R(0)
end
