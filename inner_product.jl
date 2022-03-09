using Quadrature
using ChainRulesCore
using Cubature

"""
    InnerProduct{F,S}

An struct representing a linear form on a Hilbert space as an inner product ``⟨w, x⟩``.
"""
struct InnerProduct{F}
    w::F
    solver::S
end

InnerProduct(w) = InnerProduct(w, HCubatureJL())

function inner_product(f, g; solver = HCubatureJL())
    function fun(u,_)
        f(u) * g(u)
    end
    prob = QuadratureProblem(fun, 0, 1, [])
    solve(prob, solver).u
end

function (ip::InnerProduct)(x)
    inner_product(ip.w, x, solver=ip.solver)
end

function ChainRulesCore.rrule(ip::InnerProduct, x)
    y = ip(x)
    function ip_pullback(ȳ)
        q(u) = Zygote.gradient(f -> f(u), w)
        s = inner_product(q, x)

        īp = Tangent{InnerProduct}(; w= ȳ .* s, solver=NoTangent())
        x̄ = ZeroTangent()
        return 
    end
    return y, ip_pullback
end