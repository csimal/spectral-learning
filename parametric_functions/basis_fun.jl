
"""
    BasisFun{T,F} <: ParametricFun

A type representing a linear combination of basis functions.
"""
struct BasisFun{T,F} <: ParametricFun
    a::Vector{T}
    basis::F
end

"""
    eval_fun(x, a, basis)

Evaluate a linear combination of basis functions at a point `x`.

## Arguments
* `x`: the point at which to evaluate the function
* `a`: a vector of coefficients representing the function
* `basis`: a function of the form `b(k,x)` that evaluates the `k`-th basis element at `x`.
"""
function eval_fun(x, a, basis)
    y = zero(x * a[1])
    for k in 1:length(a)
        y += a[k] * basis(k, x)
    end
    return y
end

function (bf::BasisFun)(x)
    eval_fun(x, bf.a, bf.basis)
end

function grad(bf::BasisFun, x)
    [bf.basis(k,x) for k in 1:length(bf.a)]
end

Base.:+(f::BasisFun{T,F}, g::BasisFun{T,F}) = BasisFun{T,F}(f.a+g.a, f.basis)
Base.:-(f::BasisFun{T,F}, g::BasisFun{T,F}) = BasisFun{T,F}(f.a-g.a, f.basis)
Base.zero(f::BasisFun) = BasisFun(zero(f.a), f.basis)

Base.length(f::BasisFun) = length(f.a)

Base.:*(α::Number, f::BasisFun) = BasisFun(α * f.a, f.basis)