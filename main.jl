using Flux
using Statistics
using PyCall
using MLDatasets
using Makie

const skl = pyimport("sklearn")
const skld = pyimport("sklearn.datasets")

x_moons, y_moons = skld.make_moons(noise=0.3, random_state=0)

