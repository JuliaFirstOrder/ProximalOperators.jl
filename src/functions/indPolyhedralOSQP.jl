# IndPolyhedral: OSQP implementation

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
        OSQP.setup!(mod; P=speye(n), l=l, A=sparse(A), u=u, verbose=false,
            eps_abs=1e-8, eps_rel=1e-8,
            eps_prim_inf=1e-8, eps_dual_inf=1e-8)
        new(l, A, u, mod)
    end
end

# properties

is_prox_accurate(::IndPolyhedralOSQP) = false

# constructors

IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
) where R =
    IndPolyhedralOSQP{R}(l, A, u)

IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R},
    xmin::AbstractVector{R}, xmax::AbstractVector{R}
) where R =
    IndPolyhedralOSQP([l; xmin], [A; eye(size(A, 2))], [u; xmax])

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

function (f::IndPolyhedralOSQP{R})(x::AbstractVector{R}) where R
    Ax = f.A * x
    return all(f.l .<= Ax .<= f.u) ? zero(R) : Inf
end

# prox

function prox!(y::AbstractVector{R}, f::IndPolyhedralOSQP{R}, x::AbstractVector{R}, gamma::R=one(R)) where R
    OSQP.update!(f.mod; q=-x)
    results = OSQP.solve!(f.mod)
    y .= results.x
    return zero(R)
end

# naive prox

# we want to compute the projection p of a point x
#
# primal problem is: minimize_p (1/2)||p-x||^2 + g(Ap)
# where g is the indicator of the box [l, u]
#
# dual problem is: minimize_y (1/2)||-A'y||^2 - x'A'y + g*(y)
# can solve with (fast) dual proximal gradient method

function prox_naive(f::IndPolyhedralOSQP{R}, x::AbstractVector{R}, gamma::R=one(R)) where R
    y = zeros(R, size(f.A, 1)) # dual vector
    y1 = y
    g = IndBox(f.l, f.u)
    gstar = Conjugate(g)
    gstar_y = zero(R)
    stepsize = one(R)/norm(full(f.A*f.A'))
    for it = 1:1e6
        w = y + (it-1)/(it+2)*(y - y1)
        y1 = y
        z = w - stepsize * (f.A * (f.A'*w - x))
        y, = prox(gstar, z, stepsize)
        if norm(y-w)/(1+norm(w)) <= 1e-12 break end
    end
    p = -f.A'*y + x
    return p, zero(R)
end
