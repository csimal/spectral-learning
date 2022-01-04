using Flux
using Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using MLDatasets
using Makie

if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 3e-04
    batchsize::Int = 1024
    epochs::Int = 10
    device::Function = gpu
end

function get_data(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true

    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    ytrain = onehotbatch(ytrain, 0:9)
    ytest = onehotbatch(ytest, 0:9)

    traindata = DataLoader((xtrain,ytrain), batchsize=args.batchsize, shuffle=true)
    testdata = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return traindata, testdata
end

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
        Dense(prod(imgsize), 32, relu),
        Dense(32, nclasses)
    )
end

function build_spectral_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
        Spectral(prod(imgsize), 32, relu),
        Spectral(32, nclasses)
    )
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(dataloader, model)
    acc = 0
    for (x,y) in dataloader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) * 1 / size(x,2)
    end
    acc / length(dataloader)
end

function train(; kws...)
    args = Args(; kws...)
    traindata, testdata = get_data(args)
    m = build_model()
    loss(x,y) = logitcrossentropy(m(x), y)

    eval_cb = () -> @show loss_all(traindata, m)
    opt = ADAM(args.η)

    @epochs args.epochs Flux.train!(loss, params(m), traindata, opt, cb=eval_cb)
    @show accuracy(traindata, m)
    @show accuracy(testdata, m)
end