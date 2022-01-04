using Flux

struct Spectral{F, M<:AbstractMatrix, V}
    B::M
    λ::V
    b::V
    σ::F
end

function Spectral(in::Integer, out::Integer, σ=identity;
    init=glorot_uniform,
    bias=true
    )
    B = init(out, in)
    λ = init(out)
    b = bias ? init(out) : bias
    Spectral(B, λ, b, σ)
end

#Flux.trainable(s::Spectral) = (s.λ,)

@functor Spectral

function (s::Spectral)(x::AbstractVecOrMat)
    B, λ, b, σ = s.B, s.λ, s.b, s.σ
    y = B*x
    σ.(y .- λ.*y .+ b)
end

function spectralize(d::Dense)
    in, out = size(d.weight)
    Spectral(in, out, d.σ)
end
