#=
Basinhopping.jl: The basinhopping global optimization algorithm
=#

module Basinhopping

using Random
import Base: show, minimum
import Optim: minimizer, f_calls, g_calls, h_calls, converged

export basinhopping, BasinhoppingResult, minimum, minimizer, show, BasinhoppingParams

"""
Basinhopping algorithm outcomes
- `niter_completed`: Maximum number of local optimisatiosn reached
- `early_stop`: Callback function requested an early stop to the algorithm
- `success_condition`: Same minimum found more than specified number of times
"""
@enum BasinhoppingOutcome niter_completed early_stop success_condition

"""
Result of applying acceptance test
- `accept`: Test passed, local minimum accepted
- `reject`: Test failed, local minimum rejected
- `force`: Forced acceptance of local minimum
"""
@enum Test result accept reject force

"Represents test indicating whether optimization result should be accepted"
abstract type AcceptanceTest end

"""
Metropolis acceptance test parameters
"""
struct MetropolisTest{T} <: AcceptanceTest
    beta::Float64
    rng::T
end

"""
    function MetropolisTest(T[, rng])

Create Metropolis acceptance test with temperature `T = 1/beta`.
"""
function MetropolisTest(T, rng = Random.GLOBAL_RNG)
    beta = T != 0 ? 1.0 / T : -Inf
    return MetropolisTest(beta, rng)
end

"Group multiple tests together"
struct CompositeTest <: AcceptanceTest
    list::Vector{AcceptanceTest}
end

function CompositeTest(test_list...)
    return CompositeTest([t for t in test_list])
end


"""
    function apply_test(test,
                        minimum_after_local_opt,
                        x_after_local_opt,
                        minimum,
                        x)
Apply acceptance test.

# Arguments
- `optimizer`: an optimization routine; must be a callable accepting 
                a single argument, the initial parameters for a local optimization
- `x0`: initial parameters for first local optimization
- `parameters`: optional parameters; see `BasinhoppingParams`
"""
function apply_test(test::AcceptanceTest,
                    minimum_after_local_opt,
                    x_after_local_opt,
                    minimum,
                    x)
    throw(ErrorException("Function not implemented for test $(typeof(test))"))
end


"""  
    function apply_test(test::CompositeTest,
                        minimum_after_local_opt,
                        x_after_local_opt,
                        minimum,
                        x)
Evaluates composite acceptace test; test is accepted either if
all tests are accepted or if one test forces the acceptance by
returning `force`.
"""
function apply_test(test::CompositeTest,
                    minimum_after_local_opt,
                    x_after_local_opt,
                    minimum,
                    x)
    accept_minimisation = accept
    for t in test.list
        test_result = apply_test(t, minimum_after_local_opt, x_after_local_opt, minimum, x)
        if test_result == force
            return force
        elseif test_result == reject
            accept_minimisation = reject
        end
    end
    return accept_minimisation
end

"Evaluates Metropolis criterion."
function metropolis_criterion(new_minimum, old_minimum, beta, rng)
    w = exp(min(0, -(new_minimum - old_minimum) * beta))
    rand = Random.rand(rng, Float64)
    return w >= rand ? accept : reject
end

"""  
    function apply_test(test::MetropolisTest,
                        minimum_after_local_opt,
                        x_after_local_opt,
                        minimum,
                        x)
Evaluates Metropolis acceptace test.

See also [`metropolis_criterion`](@ref)
"""
function apply_test(test::MetropolisTest,
                    minimum_after_local_opt,
                    x_after_local_opt,
                    minimum,
                    x)
    return metropolis_criterion(minimum_after_local_opt, minimum, test.beta, test.rng)
end

"""Represents perturbation of parameters between local optimizations."""
abstract type StepTaker end

"""
Random displacement of parameters.

# Fields
- `stepsize::Float64`: Size of random displacement.
- `rng::RNG`: Source of pseudorandomness.
- `interval::Int64`: total number of function evaluations.
- `target_accept_rate::Float64`: Ratio between local minima being accepted and all local optimizations.
- `factor::Float64`: Factor by which to adjust step size when updating.
"""
mutable struct RandomDisplacement{RNG} <: StepTaker
    stepsize::Float64
    rng::RNG
    interval::Int64
    target_accept_rate::Float64
    factor::Float64
end

"""
    take_step!(d::StepTaker, x)

Perturbs parameters `x` as specified by a step taker `d`
"""
function take_step!(d::StepTaker, x) end

"""
    take_step!(d::RandomDisplacement, x)

Random perturbation in every coordinate of `x`.
"""
function take_step!(d::RandomDisplacement, x)
    x[:] += rand(d.rng, Float64, size(x)) .* 2 .* d.stepsize .- d.stepsize
end

"""
    update!(d::StepTaker, nstep, naccept)

Updates step taking routine.

# Arguments
- `d::StepTaker`: 
- `nstep::Int`: Number of local optimizations performed so far.
- `naccept::Int`: Number of local minima accepted so far.
"""
function update!(d::StepTaker, nstep::Int, naccept::Int) end

"""
    update!(d::RandomDisplacement, nstep, naccept)
"""
function update!(d::RandomDisplacement, nstep::Int, naccept::Int)
    if (nstep % d.interval) == 0
        accept_rate = naccept / nstep
        if accept_rate > d.target_accept_rate
            # We're accepting too many steps.  This generally means we're
            # trapped in a basin. Take bigger steps
            d.stepsize /= d.factor
        else
            # We're not accepting enough steps. Take smaller steps
            d.stepsize *= d.factor
        end
    end
end

"""
Basinhopping algorithm result

# Fields
- `minimization_result`: local optimization result identified as global minimum
- `total_iterations::Int64`: total number of local optimizations performed
- `f_calls::Int64`: total number of function evaluations
- `g_calls::Int64`: total number of gradient calls
- `h_calls::Int64`: total number of Hessian evaluations
- `exit_code::BasinhoppingOutcome`: Result of algorithm (see `BasinhoppingOutcome` documentation)
"""
struct BasinhoppingResult{MinimizationResult}
    minimization_result::MinimizationResult
    total_iterations::Int64
    f_calls::Int64
    g_calls::Int64
    h_calls::Int64
    exit_code::BasinhoppingOutcome
end

"""
    struct BasinhoppingParams{T<:StepTaker,U<:AcceptanceTest,C<:Function}

Optional parameters to Basinhopping algorithm

# Fields
- `niter::Int64`: Total number of local optimisations performed.
- `step_taker::T`: Total number of local optimizations performed.
- `test::U`: Test determining whether local optimization is accepted.
- `callback::C`: Function to be called after each local optimization. Signature: `(x, minimum, parameters_after_local_opt, minimum_after_local_opt, test_result) -> bool`. If this returns `true` the algorithm is stopped.
- `niter_success::Int64`: Stop if the global minimum candidate remains the same for this number of iterations.
"""
struct BasinhoppingParams{T<:StepTaker,U<:AcceptanceTest,C<:Function}
    niter::Int64
    step_taker::T
    test::U
    callback::C
    niter_success::Union{Int64, Nothing}
end

# Default parameters
default_step_taker = RandomDisplacement(0.5, Random.GLOBAL_RNG, 50, 0.5, 0.9)
default_test = MetropolisTest(1.0, Random.GLOBAL_RNG)
default_callback = (x, energy, x_after_quench, energy_after_quench, accept) -> false
default_niter_success = nothing

"""
    function BasinhoppingParams(;<keyword arguments>)

# Keyword arguments
- `niter`: Default: `20`.
- `step_taker`: Default `RandomDisplacement(0.5, Random.GLOBAL_RNG, 50, 0.5, 0.9)`.
- `test`: Default: `MetropolisTest(1.0, Random.GLOBAL_RNG)`.
- `callback::Int64`: Default: `(...) -> false`. 
- `niter_success::Int64`: Default: `2`
"""
function BasinhoppingParams(;
    niter = 20,
    step_taker = default_step_taker,
    test = default_test,
    callback = default_callback,
    niter_success = default_niter_success,
)
    return BasinhoppingParams(niter, step_taker, test, callback, niter_success)
end

"""
    function basinhopping(optimizer, x0[, parameters])

Basinhopping optimization algorithm.

# Arguments
- `optimizer`: an optimization routine; must be a callable accepting 
               a single argument, the initial parameters for a local optimization
- `x0`: initial parameters for first local optimization
- `parameters`: optional parameters; see `BasinhoppingParams`
"""
function basinhopping(optimizer, x0, parameters::BasinhoppingParams = BasinhoppingParams())
    niter = parameters.niter
    step_taker = parameters.step_taker
    test = parameters.test
    callback = parameters.callback
    niter_success = parameters.niter_success

    if niter_success === nothing
        niter_success = niter + 2
    end

    minimization_failures = 0
    nstep = 0
    naccept = 0

    minres = optimizer(x0)
    x = minimizer(minres)
    min = minimum(minres)

    total_nfev = f_calls(minres)
    total_njev = g_calls(minres)
    total_nhev = h_calls(minres)

    if !converged(minres)
        minimization_failures += 1
    end

    nr_of_iterations_since_last_minimum = 0
    exit_code = niter_completed

    x_after_step = empty(x)

    for i = 1:niter
        nstep += 1
        
        # periodically update parameter displacement
        #update!(step_taker, nstep, naccept)

        # displace parameters using provided method
        copy!(x_after_step, x)
        take_step!(step_taker, x_after_step)

        # perform a local minimization
        new_minres = optimizer(x_after_step)

        # check if local minimization was successful
        if !converged(new_minres)
            minimization_failures += 1
        end

        # extract local minimum and parameters at which it was found
        new_x = minimizer(new_minres)
        new_min = minimum(new_minres)
        
        # track number of function, gradient, hessian evaluations
        total_nfev += f_calls(new_minres)
        total_njev += g_calls(new_minres)
        total_nhev += h_calls(new_minres)

        # apply provided acceptance test
        test_result = apply_test(test, new_min, new_x, min, x)

        val = callback(x, min, new_x, new_min, test_result)
        if val == true
            break
        end

        if test_result == accept
            naccept += 1
            min = new_min
            x = new_x
            # track lowest minimum found 
            if new_min < minimum(minres)
                minres = new_minres
                nr_of_iterations_since_last_minimum = 0
            else
                nr_of_iterations_since_last_minimum += 1
                # stop algorithm if no new minimum is found 
                if nr_of_iterations_since_last_minimum > niter_success
                    exit_code = success_condition
                    break
                end
            end
        end

    end

    return BasinhoppingResult(minres, nstep, total_nfev, total_njev, total_nhev, exit_code)
end

"Reports on exit code of `basinhopping`"
function success_string(self::BasinhoppingOutcome)
    basinhopping_outcomes = Dict{BasinhoppingOutcome,String}(
        niter_completed =>
            "requested number of basinhopping iterations completed successfully",
        early_stop => "callback function requested stop early by returning true",
        success_condition => "success condition satisfied",
    )
    return basinhopping_outcomes[self.code]
end

"Returns global minimum found by basinhopping"
function minimum(result::BasinhoppingResult)
    return minimum(result.minimization_result)
end

"Returns parameters at which global minimum was found"
function minimizer(result::BasinhoppingResult)
    return minimizer(result.minimization_result)
end

"Returns total number of function evaluations"
function f_calls(result::BasinhoppingResult)
    return result.f_calls
end

"Returns total number of gradient evaluations"
function g_calls(result::BasinhoppingResult)
    return result.g_calls
end

"Returns total number of Hessian evaluations"
function h_calls(result::BasinhoppingResult)
    return result.h_calls
end

"Show global minimum found"
function show(result::BasinhoppingResult)
    show(result.minimization_result)
end

end # module
