function isapprox_le(x::Number, y::Number; atol::Real=0, rtol::Real=Base.rtoldefault(x,y,atol))
    x <= y || (isfinite(x) && isfinite(y) && abs(x-y) <= max(atol, rtol*max(abs(x), abs(y))))
end
