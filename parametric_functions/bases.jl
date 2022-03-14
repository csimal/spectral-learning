
function chebyshev_basis(k,x)
    if abs(x) < 1
        cos(k * acos(x))
    elseif x >= 1
        cosh(k * acosh(x))
    else
        (-1)^k * cosh(k * acosh(-x))
    end
end

chebyshev_basis(k,x::AbstractVector) = chebyshev_basis.(k,x)
