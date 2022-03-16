using ApproxFun
using Quadrature
using Flux
import ChainRulesCore: rrule, frule

struct SpectralKernelOperator{F,L,F1,F2,P}
    B::L # Linear Operator
    λ::F1 # Spectral Scaling Function
    b::F2 # Bias function
    σ::F # activation function
    project::P # Projection operator
end

SpectralKernelOperator(B, λ, b) = SpectralKernelOperator(B, λ, b, identity, f -> Fun(f, 0..1))

Flux.@functor SpectralKernelOperator

function (sk::SpectralKernelOperator)(x)
    B, λ, b, σ = sk.B, sk.λ, sk.b, sk.σ
    y = B(x)
    z = t -> let yt = y(t)
        σ(yt - λ(t)*yt + b(t))
    end
    sk.project(z)
end

function ChainRulesCore.rrule(sk::SpectralKernelOperator, x)
    y = sk(x)
    function sk_pullback(δy)
        
    end
    return y, sk_pullback
end
