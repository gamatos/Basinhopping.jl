# Basinhopping.jl

## Description

Julia implementation of the basinhopping global optimization algorithm. This algorithm attempts to find the global minimum of a given function by successively performing a local optimization followed by a perturbation of the optimal parameters. This allows the algorithm to "jump" between local minima until the global one is found. An acceptance test determines whether a "jump" is performed or not.

## Tutorial

### Basic usage

```julia
using Optim
using Basinhopping
using LinearAlgebra

f(x) = norm(x)

# Define local optimiser to use
opt = (initial_guess)->optimize(f, initial_guess, LBFGS())

# Optimise
ret = basinhopping(opt, [1.0, 1.0], BasinhoppingParams(niter=200))
```

Optional arguments are passed using the `BasinhoppingParams` constructor. The keyword arguments are
- `niter`: Integer specifying how many local optimisations should be performed.
- `step_taker`: How to perturb local minimum parameters to find initial condition for next local optimization (see [Perturbation of optimal parameters](#perturbation-of-optimal-parameters)).
- `test`: Determines how to decide whether to accept a local minimum or to discard it (see [Acceptance criteria](#acceptance-criteria)).
- `callback`: Function that gets called after each local optimization. Has signature `(x, min, new_x, new_min, test_result) -> bool`. Returning `true` stops the algorithm.
- `niter_success`: If more than this number of tests are rejected, return current minimum as the global minimum. Defaults to going through all iterations regardless of number of rejections.

The optimizer supplied must return a structure which has the following functions defined: `minimizer`, `minimum`, `f_calls`, `g_calls`, `h_calls`. This follows the interface of interface `Optim.jl`, which is supported as a provider of local optimizers.

### Acceptance criteria

After a local minimum is found, it is compared to the previously found minimum using an acceptance test. The test is specified by passing an instance of a subtype of `AcceptanceTest` to the `test` keyword argument of `BasinhoppingParams`. 

The default acceptance test is a Metropolis test, and is specified by the `MetropolisTest` structure. The test is accepted if the new minimum has a lower value than the previous one; otherwise it is accepted with probability

```
exp(-(new_minimum - old_minimum) / T)
```
where `T` (the temperature) is passed as a parameter to the structure.

To implement a custom acceptance criterion, create a subtype of `AcceptanceTest` and implement the `take_step!(d::StepTaker, x)` function for that type, where `x` are the coordinates of the current minimum.


### Perturbation of optimal parameters

After a local minimum is found, the coordinates of that minimum are perturbed and a new local optimization is performed. The method by which that perturbation is made is defined by passing an instance of a subtype of `StepTaker` to the `step_taker` keyword argument of `BasinhoppingParams`. The default perturbation is a uniform random perturbation in every direction in the coordinates, and is specified by the `RandomDisplacement` structure.

To implement a custom perturbation, create a subtype of `StepTaker` and implement
* A `take_step!(d::StepTaker, x)` function for that type, where `x` are the coordinates of the current minimum.
* An `update!(d::StepTaker, nstep::Int, naccept::Int)` function, where `nstep` is the number of local optimisations performed so far and `naccept` is the number of those which were accepted by the acceptance test. This function gets called between iterations and allows one to update the parameters of the step taker dynamically.

## Installation

The package can be installed with Pkg.add
```julia
julia> using Pkg; Pkg.add("Basinhopping")
```

or through the pkg REPL mode by typing
```
] add Basinhopping
```
## References

1. Wales, David J. 2003, Energy Landscapes, Cambridge University Press, Cambridge, UK.
2. Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms. Journal of Physical Chemistry A, 1997, 101, 5111.
3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

