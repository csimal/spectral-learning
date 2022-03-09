using ApproxFun
using BasisFunctionExpansions
using SmoothingSplines
#using Wavelets
#using ContinuousWavelets
using GLMakie
#using ElectronDisplay
using ForwardDiff
using Zygote


GLMakie.inline!(false)

include("kernel_operators.jl")
include("inner_product.jl")
include("continuous_spectral_layers.jl")

f(x) = sin(4π*x) + cos(5π*x)

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

p = Polynomial(1:5)

Zygote.gradient(p -> p(2), p)

ip = InnerProduct(p)
ip(f)
Zygote.gradient(ip -> ip(f), ip)


f_app = Fun(f, 0..1)

x = range(0.0,1.0, length=100)

rbf = UniformRBFE(x, 10)
bfa = BasisFunctionApproximation(f.(x),x,rbf,1)

spl = fit(SmoothingSpline, x, f.(x), 10.0)

lines(x, f.(x), label="True f")
lines!(x, f_app.(x), label="Chebyshev")
lines!(x, bfa(x), label="RBF")
lines!(x, [SmoothingSplines.predict(spl, y) for y in x], label="Smoothing Splines")
axislegend()
current_figure()

f_app(0.5)
bfa([0.5])
SmoothingSplines.predict(spl, 0.5)

ko = KernelOperator(FourierKernel(10.0,0.0))

ko.kernel(0.5,0.5)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ax,x,x,ko.kernel.(x,x'))
Colorbar(fig[1,2], hm)
fig

g = ko(f)

u = g(0.5)

g_app = Fun(g, 0..1)

g_app(0.5)

lines(x, g.(x), label="True g")
lines!(x, g_app.(x), label="Chebyshev")
axislegend()
current_figure()

cs = SpectralKernelOperator(ko, cos, sin)

h = cs(f_app)

lines(x, h.(x))