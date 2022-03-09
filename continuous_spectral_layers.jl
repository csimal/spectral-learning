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

function (cs::SpectralKernelOperator)(x)
    B, λ, b, σ = cs.B, cs.λ, cs.b, cs.σ
    y = B(x)
    z = t -> let yt = y(t)
        σ(yt - λ(t)*yt + b(t))
    end
    cs.project(z)
end

#function ChainRulesCore.rrule(cs::ContinuousSpectral, x::Function)
    
#end

struct InnerProductOperator{F,W,B}
    w::Vector{W}
    b::B
    σ::F
end

Flux.@functor InnerProductOperator

function (ipo::InnerProductOperator)(x)
    y = [w(x) for w in ipo.w]
    σ.(y + b)
end