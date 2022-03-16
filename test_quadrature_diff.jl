using Quadrature, ForwardDiff, FiniteDiff, Zygote
using Cuba, Cubature, HCubature
using ChainRulesCore

f(x,p) = sum(sin.(x .* p))

lb = ones(2)
ub = 3ones(2)
p = [1.5,2.0]

function testf(p)
    prob = QuadratureProblem{false}(f,lb,ub,p)
    sin(solve(prob,CubaCuhre(),reltol=1e-6,abstol=1e-6)[1])
end
dp1 = Zygote.gradient(testf,p)
dp2 = FiniteDiff.finite_difference_gradient(testf,p)
dp3 = ForwardDiff.gradient(testf,p)
dp1[1] ≈ dp2 ≈ dp3

struct Polynomial
    a::Vector{Float64}
end

function (p::Polynomial)(x)
    y = 0.0
    for α in p.a
        y *= x
        y += α
    end
    return y
end

Base.length(p::Polynomial) = length(p.a)

function testp(p)
    function g(u,q)
        p(u) * identity(u)
    end
    prob = QuadratureProblem(g,0,1,[])
    solve(prob, CubatureJLh()).u
end

function ChainRulesCore.rrule(::typeof(testp), p)
    y = testp(p)
    function testp_pullback(ȳ)
        q(x) = Zygote.gradient(p -> p(x), p)[1][1]
        function fun(u,_)
            q(u) * identity(u)
        end
        prob = QuadratureProblem(fun, 0,1, [])
        sol = solve(prob, HCubatureJL())
        
        p̄ = Tangent{Polynomial}(; a= ȳ .* sol.u)
        return NoTangent(), p̄
    end
    return y, testp_pullback
end

p = Polynomial(1:5)

testp(p)

Zygote.gradient(p, 2)
Zygote.gradient(p -> p(2), p)[1][1]
Zygote.gradient(testp, p)
ForwardDiff.gradient(testp, p)

q(x) = Zygote.gradient(p -> p(x), p)[1][1]
q(1)
function fun(u,_)
    q(u) .* sin(3*u)
end
prob = QuadratureProblem(fun, 0,1, [])
sol = solve(prob, HCubatureJL())

Zygote.gradient(f -> f(2), sin)

function eval_poly(a::Vector{<:Number}, x)
    y = zero(x * a[1])
    for α in a
        y *= x
        y += α
    end
    return y
end

eval_poly(p::Polynomial, x) = eval_poly(p.a, x)

Zygote.gradient(p -> eval_poly(p,2), p)

function teste(p::Polynomial)
    function fun(u,_)
        eval_poly(p,u) * identity(u)
    end
    prob = QuadratureProblem(fun, 0, 1, ())
    solve(prob, HCubatureJL()).u
end

function testa(a::Vector)
    function fun(u,p)
        eval_poly(p, u) * identity(u)
    end
    prob = QuadratureProblem(fun, 0.0, 1.0, a)
    solve(prob, HCubatureJL()).u
end

teste(p)
Zygote.gradient(teste, p)
ForwardDiff.gradient(teste, p)

a = collect(1:5)

testa(a)

Zygote.gradient(testa, a)
b = ForwardDiff.gradient(testa, a)
ForwardDiff.gradient(p -> eval_poly(p, 0.5), a)
Zygote.gradient(p -> eval_poly(p, 0.5), a)