using LinearAlgebra: BlasInt, chkstride1
using LinearAlgebra.LAPACK: chklapackerror
using LinearAlgebra.BLAS: @blasfunc

"""
    dspev!(jobz::Symbol, uplo::Symbol, x::StridedVector{Float64})

Computes all the eigenvalues and optionally the eigenvectors of a real
symmetric `n×n` matrix `A` in packed storage. Will corrupt `x`.

Arguments:

  `jobz`: `:N` if only eigenvalues, `:V` if eigenvalues and eigenvectors

  `uplo`: `:L` if lower triangle of `A` is stored, `:U` if upper

  `x`: `A` represented as vector of the lower (upper) n*(n+1)/2 elements, packed columnwise.

Returns:

  `W,Z` if `jobz == :V` or: `W` if `jobz == :N` such that `A=Z*diagm(W)*Z'`
"""

function dspev!(jobz::Symbol, uplo::Symbol, A::StridedVector{Float64})
    chkstride1(A)
    vecN = length(A)
    n = try
        Int(sqrt(1/4+2*vecN)-1/2)
    catch
        throw(DimensionMismatch("A has length $vecN which is not N*(N+1)/2 for any integer N"))
    end
    W     = similar(A, Float64, n)
    Z     = similar(A, Float64, n, n)
    work  = Array{Float64}(undef, 1)
    lwork = BlasInt(3*n)
    info  = Ref{BlasInt}()
    work = Array{Float64}(undef, lwork)
    ccall((@blasfunc(dspev_), Base.liblapack_name), Cvoid,
          (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{Float64},
          Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
          jobz, uplo, Ref(n), A,
          W, Z, Ref(n), work, info)
    chklapackerror(info[])
    jobz == :V ? (W, Z) : W
end

function dspevV!(uplo::Symbol, A::StridedVector{Float64})
    jobz = :V
    chkstride1(A)
    vecN = length(A)
    n = try
        Int(sqrt(1/4+2*vecN)-1/2)
    catch
        throw(DimensionMismatch("A has length $vecN which is not N*(N+1)/2 for any integer N"))
    end
    W     = similar(A, Float64, n)
    Z     = similar(A, Float64, n, n)
    work  = Array{Float64}(undef, 1)
    lwork = BlasInt(3*n)
    info  = Ref{BlasInt}()
    work = Array{Float64}(undef, lwork)
    ccall((@blasfunc(dspev_), Base.liblapack_name), Cvoid,
          (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{Float64},
          Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
          jobz, uplo, Ref(n), A,
          W, Z, Ref(n), work, info)
    chklapackerror(info[])
    return W, Z
end
