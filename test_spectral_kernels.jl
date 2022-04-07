using Flux
using ApproxFun
using GLMakie
using SpectralLearning
using SpectralLearning.ParametricFunctions
using ForwardDiff
using Zygote

sno = SpectralNeuralOperator(
    KernelOperator(FourierKernel()),
    BasisFun(collect(1:10), ParametricFunctions.Chebyshev()),
    BasisFun(collect(1:10), ParametricFunctions.Chebyshev()),
)


f = sin

g = sno(f)

x = 0.0:0.01:1.0


lines(x, g.(x), label="primal output")
lines!(x, g_.(x), label="pullback output")
axislegend()
current_figure()

g_, pb = Zygote.pullback(sno, f)

Zygote.pushforward(sno, t -> 0.1)

δsno = pb(sin)

model = Chain(
    sno,
    SpectralLearning.InnerProduct(cos)
)

model(sin)

Flux.gradient(model, sin)
Zygote.gradient(model, sin)
Flux.gradient(m -> m(sin), model)