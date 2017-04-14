import Base.SparseArrays.SPQR: ORDERING_DEFAULT, DEFAULT_TOL
using Base.SparseArrays.CHOLMOD: C_Sparse, Sparse, VTypes, ITypes, common, SuiteSparse_long

# # Optional type if we want to work inside SuiteSparse
# type QRQless{Tv<:VTypes} <: Base.LinAlg.Factorization{Tv}
#     m::Int
#     n::Int
#     R::Ref{Ptr{C_Sparse{Tv}}}
#     E::Ref{Ptr{Clong}}
#     function QRQless{Tv}(m::Integer, n::Integer, R::Ref{Ptr{C_Sparse{Tv}}}, E::Ref{Ptr{Clong}}) where Tv<:VTypes
#         if R.x == C_NULL || E.x == C_NULL
#             throw(ArgumentError("factorization failed for unknown reasons. Please submit a bug report."))
#         end
#         new(m, n, R, E)
#     end
# end

""" QlessQR{Tv} <: Base.LinAlg.Factorization{Tv}
Q-less QR factorization of matrix A, so that QR=AP, where Q unitary, R upper triangular, P permutation matrix
elements: R sparse upper traiangular
          p permutation vector
"""
immutable QlessQR{Tv<:VTypes} <: Base.LinAlg.Factorization{Tv}
    m::Int
    n::Int
    R::UpperTriangular{Tv,SparseMatrixCSC{Tv,Clong}}
    p::Array{Clong,1}
    function QlessQR{Tv}(m::Integer, n::Integer, R::Ref{Ptr{C_Sparse{Tv}}}, E::Ref{Ptr{Clong}}) where Tv<:VTypes
        if R.x == C_NULL || E.x == C_NULL
            throw(ArgumentError("factorization failed for unknown reasons."))
        end
        R2 = sparse(Sparse(R.x))                  #Create SparseMatrixCSC from pointer
        p = unsafe_wrap(Array{Clong,1}, E.x, n)   #Get permutation from pointer
        p .+= one(Clong)                          #Convert to start at index 1
        new(m, n, UpperTriangular(R2), p)
    end
end

"""F = qrfactqless(A::SparseMatrixCSC{Tv, Ti}), where F::QRQless
Do a Q-less QR=AE factorization of the sparse matrix A.
Settings:
  Automatic ordering
  Economic size: R is min(m,n)-by-n matrix
"""
function qrfactqless{Tv<:VTypes, Ti<:ITypes}(A::SparseMatrixCSC{Tv, Ti})
    getCTX = Cint(1)
    qrfactqless(Sparse(A,0), ORDERING_DEFAULT, DEFAULT_TOL, getCTX)
end

function qrfactqless{Tv<:VTypes}(A::Sparse{Tv}, ordering::Integer, tol::Real, getCTX::Cint)
    s = unsafe_load(A.p)
    if s.stype != 0
        throw(ArgumentError("stype must be zero"))
    end

    n = size(A)[2]                #                     , m-by-n sparse matrix
    econ = Clong(n)               # Long econ           , e = max(min(m,econ),rank(A))
    Er = Ref{Ptr{Clong}}()        # SuiteSparse_long **E, permutation of 0:n-1, NULL if identity
    Rf = Ref{Ptr{C_Sparse{Tv}}}() # cholmod_sparse **R  , e-by-n sparse matrix
    f = ccall((:SuiteSparseQR_C, :libspqr), SuiteSparse_long,
         #Types:
        (Cint, Cdouble, Clong, Cint,
         Ptr{Sparse{Tv}}, Ptr{Void},              Ptr{Void},        Ptr{Void},
         Ptr{Void},       Ref{Ptr{C_Sparse{Tv}}}, Ref{Ptr{Clong}},  Ptr{Void},
         Ptr{Void},       Ptr{Void},              Ptr{Void}),
         #Values:
         ordering,        tol,                    econ,             getCTX,
         get(A.p),        C_NULL,                 C_NULL,           C_NULL,
         C_NULL,          Rf,                     Er,               C_NULL,
         C_NULL,          C_NULL,                 common())
    QlessQR{Tv}(size(A)..., Rf, Er)
end

function Base.getindex(F::QlessQR, s::Symbol)
  if s == :R
    return F.R
  elseif s == :p
    return F.p
  elseif s == :P
    #This matrix allows solving projections with
    # y = x-A'*P*(R\(R'\(P'*res)))
    return permute(speye(F.n), s.p, 1:F.n)
  else
    throw(ArgumentError("Invalid argument $s. Should be either :R, :p"))
  end
end
