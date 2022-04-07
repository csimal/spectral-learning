using Pkg

packages = [
    "https://github.com/csimal/SpectralLearning.jl.git",
    "Flux",
    "ApproxFun",
    "NeuralOperators",
    "Quadrature",
    "MLDatasets",
    "KernelFunctions",
    "Plots"
]

Pkg.add(packages)