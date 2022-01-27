using ApproxFun
using Quadrature

abstract type AbstractOperator end

struct KernelOperator{K,S} <: AbstractOperator
    kernel::K
    solver::S = QuadGKJL()
end

function (ko::KernelOperator)(f::Function; solver=k.solver)
    k = ko.kernel
    fun = function(u,p) 
        k(p,u) * f(u)
    end
    return function(x)
        prob = QuadratureProblem(fun,0,1,x)
        solve(prob, solver)
    end
end

struct FourierKernel{T<:Real}
    ω::T
    θ::T
end

FourierKernel(; ω₀ = 0.0, ω₁ = 1.0) = FourierKernel((ω₁-ω₀), ω₀)

(ff::FourierKernel)(x,y) = sin(ff.ω*π*x*y + ff.θ)

struct PolynomialKernel{M<:AbstractMatrix}
    m::M
end

function (po::PolynomialOperator)(x,y)
    (m,n) = size(po.m)
    X = [x^(k-1) for k in 1:m]
    Y = [y^(k-1) for k in 1:n]
    X * po.m * Y
end

struct ContinuousSpectral{F,M,V}
    B::M
    λ::V
    b::V
    σ::F
end