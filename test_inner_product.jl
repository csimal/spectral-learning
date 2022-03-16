using ApproxFun
using Quadrature, HCubature
using Zygote

include("parametric_functions/ParametricFunctions.jl")
#import .ParametricFunctions as PF
#using .ParametricFunctions

include("inner_product.jl")

x = Fun(sin, 0..1)

w = Fun(t -> t^2, 0..1)

w = BasisFun(collect(1.0:5.0), Chebyshev())

ip = InnerProduct(w)

y = ip(x)

Zygote.gradient(p -> p(0.5), w)

Zygote.gradient(p -> p(x), ip)


