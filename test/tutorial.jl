using NLopt, Test
using ForwardDiff

count = 0 # keep track of # function evaluations

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end

    global count
    count::Int += 1
    println("f_$count($x)")

    sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4
opt.min_objective = myfunc
opt.inequality_constraint = (x,g) -> myconstraint(x,g,2,0)
opt.inequality_constraint = (x,g) -> myconstraint(x,g,-1,1)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
println("got $minf at $minx after $count iterations (returned $ret)")

@test minx[1] ≈ 1/3 rtol=1e-5
@test minx[2] ≈ 8/27 rtol=1e-5
@test minf ≈ sqrt(8/27) rtol=1e-5
@test ret == :XTOL_REACHED
@test opt.numevals == count


## a helpful way to organize the objective function with ForwardDiff
solution_log = "./log_solutions.txt"

function myfunc_1(x::Vector)
    y = sqrt(x[2])
    return y
end

function myfunc_2(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[:] = ForwardDiff.gradient(myfunc_1, x) # in-place evaluation required.
    end
    if iszero(maximum(isnan.(grad))
            grad_max = findmax(abs.(grad))
            grad_val = grad_max[1]
            grad_ind = grad_max[2]
            println("    Max. abs. gradient at $grad_ind with value $grad_val.")
    else
            println("    Gradient has NaN elements.")
            # We could switch finite differecing here, 
            # but sometimes it's just quicker to let the current evaluation fail.
    end
        
    global count
    count::Int += 1
    println("f_$count($x)")

    y = sqrt(x[2])
    
    # record evaluations
    open(solution_log, "a") do f
            println(f, y, "    ", x')
    end
    
    return y
end
