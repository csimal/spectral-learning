using Flux

struct SIRENLayer{T <: AbstractMatrix}

end

function siren_uniform(rng=GLOBAL_RNG, dims...)
    fan_in = dims[1]
    x = sqrt(6/fan_in)
    u = rand(rng, Float32, dims)
    -x .+ u .* (2x)
end

SIREN(in::Integer, out::Integer, bias = true) = Dense(in, out, sin, init = siren_uniform)