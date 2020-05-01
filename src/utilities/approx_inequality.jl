# This is adapted from Base.isapprox
# https://github.com/JuliaLang/julia/blob/381693d3dfc9b7072707f6d544f82f6637fc5e7c/base/floatfuncs.jl#L222-L291
function isapprox_le(x::Number, y::Number; atol::Real=0, rtol::Real=Base.rtoldefault(x,y,atol))
    x <= y || (isfinite(x) && isfinite(y) && abs(x-y) <= max(atol, rtol*max(abs(x), abs(y))))
end
