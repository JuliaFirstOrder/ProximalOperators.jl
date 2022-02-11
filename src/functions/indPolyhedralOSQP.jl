# IndPolyhedral: OSQP implementation

using OSQP

struct IndPolyhedralOSQP{R} <: IndPolyhedral
    l::AbstractVector{R}
    A::AbstractMatrix{R}
    u::AbstractVector{R}
    mod::OSQP.Model
    function IndPolyhedralOSQP{R}(
        l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
    ) where R
        m, n = size(A)
        mod = OSQP.Model()
        if !all(l .<= u)
            error("function is improper (are some bounds inverted?)")
        end
        OSQP.setup!(mod; P=SparseMatrixCSC{R}(I, n, n), l=l, A=sparse(A), u=u, verbose=false,
            eps_abs=eps(R), eps_rel=eps(R),
            eps_prim_inf=eps(R), eps_dual_inf=eps(R))
        new(l, A, u, mod)
    end
end

# properties

is_prox_accurate(::Type{<:IndPolyhedralOSQP}) = false

# constructors

IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
) where R =
    IndPolyhedralOSQP{R}(l, A, u)

IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R},
    xmin::AbstractVector{R}, xmax::AbstractVector{R}
) where R =
    IndPolyhedralOSQP([l; xmin], [A; I], [u; xmax])

IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, args...
) where R =
    IndPolyhedralOSQP(
        l, SparseMatrixCSC(A), R(Inf).*ones(R, size(A, 1)), args...
    )

IndPolyhedralOSQP(
    A::AbstractMatrix{R}, u::AbstractVector{R}, args...
) where R =
    IndPolyhedralOSQP(
        R(-Inf).*ones(R, size(A, 1)), SparseMatrixCSC(A), u, args...
    )

# function evaluation

function (f::IndPolyhedralOSQP)(x)
    R = eltype(x)
    Ax = f.A * x
    return all(f.l .<= Ax .<= f.u) ? R(0) : Inf
end

# prox

function prox!(y, f::IndPolyhedralOSQP, x, gamma)
    R = eltype(x)
    OSQP.update!(f.mod; q=-x)
    results = OSQP.solve!(f.mod)
    y .= results.x
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

function prox_naive(f::IndPolyhedralOSQP, x, gamma)
    R = eltype(x)
    y = zeros(R, size(f.A, 1)) # dual vector
    y1 = y
    g = IndBox(f.l, f.u)
    gstar = Conjugate(g)
    gstar_y = R(0)
    stepsize = R(1)/opnorm(Matrix(f.A*f.A'))
    for it = 1:1e6
        w = y + (it-1)/(it+2)*(y - y1)
        y1 = y
        z = w - stepsize * (f.A * (f.A'*w - x))
        y, = prox(gstar, z, stepsize)
        if norm(y-w)/(1+norm(w)) <= 1e-12 break end
    end
    p = -f.A'*y + x
    return p, R(0)
end
