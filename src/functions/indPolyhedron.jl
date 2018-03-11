export IndPolyhedron

abstract type IndPolyhedron <: ProximableFunction end

is_convex(::IndPolyhedron) = true
is_set(::IndPolyhedron) = true

IndPolyhedron(args...) = IndPolyhedronOSQP(args...)

# OSQP implementation

immutable IndPolyhedronOSQP{R} <: IndPolyhedron
    mod::OSQP.Model
    function IndPolyhedronOSQP{R}(
        l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
    ) where R
        m, n = size(A)
        mod = OSQP.Model()
        if !all(l .<= u)
            error("function is improper, should be l <= u")
        end
        OSQP.setup!(mod; P=speye(n), l=l, A=sparse(A), u=u, verbose=false,
            eps_abs=1e-8, eps_rel=1e-8,
            eps_prim_inf=1e-8, eps_dual_inf=1e-8)
        new(mod)
    end
end

IndPolyhedronOSQP(l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}) where R =
    IndPolyhedronOSQP{R}(l, A, u)

IndPolyhedronOSQP(l::AbstractVector{R}, A::AbstractMatrix{R}) where R =
    IndPolyhedronOSQP{R}(l, SparseMatrixCSC(A), R(Inf).*ones(R, size(A, 1)))

IndPolyhedronOSQP(A::AbstractMatrix{R}, u::AbstractVector{R}) where R =
    IndPolyhedronOSQP{R}(R(-Inf).*ones(R, size(A, 1)), SparseMatrixCSC(A), u)

IndPolyhedronOSQP(l::R, A::AbstractMatrix{R}, args...) where R =
    IndPolyhedronOSQP(l.*ones(R, size(A, 1)), A, args...)

IndPolyhedronOSQP(l::AbstractVector{R}, A::AbstractMatrix{R}, u::R) where R =
    IndPolyhedronOSQP(l, A, u.*ones(R, size(A, 1)))

function prox!(y::AbstractVector{R}, f::IndPolyhedronOSQP{R}, x::AbstractVector{R}, gamma::R=one(R)) where R
    OSQP.update!(f.mod; q=-x)
    results = OSQP.solve!(f.mod)
    y .= results.x
    return zero(R)
end
