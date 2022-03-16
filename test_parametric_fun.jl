
include("parametric_functions/ParametricFunctions.jl")
import .ParametricFunctions as PF

f = PF.BasisFun(collect(1.0:5.0), PF.Chebyshev())


f(0.5)
PF.grad(f, 0.5)


g = PF.from_grad(f, rand(5))

g(0.5)
PF.grad(g, 0.5)

f + g
