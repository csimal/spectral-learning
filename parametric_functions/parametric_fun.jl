using ApproxFun
using ChainRulesCore

abstract type ParametricFun <: Function end


"""
    grad(f::ParametricFun)

Return the gradient of `f` relative to its parameters.
"""
function grad end



function ChainRulesCore.rrule(typeof(eval_fun), x, a, basis)
    y = eval_fun(x, a, basis)
    function eval_fun_pullback(δy)
        δf = NoTangent()
        δx = @thunk ForwardDiff.derivative(t -> eval_fun(t, a, basis), x) * δy
        δa = [basis(k,x) for k in 1:length(a)] * δy
        return δf, δx, δa, NoTangent()
    end
    return y, eval_fun_pullback
end