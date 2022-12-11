using Basinhopping
using Test

@testset "Basinhopping.jl" begin
    using Optim

    function _test_func2d_nograd(x)
        f = (cos(14.5 * x[1] - 0.3) + (x[2] + 0.2) * x[2] + (x[1] + 0.2) * x[1] + 1.010876184442655)
        return f
    end

    function _test_func2d(x)
        f = (cos(14.5 * x[1] - 0.3) + (x[1] + 0.2) * x[1] + cos(14.5 * x[2] -
            0.3) + (x[2] + 0.2) * x[2] + x[1] * x[2] + 1.963879482144252)
        return f
    end

    function _test_func2d_grad!(storage, x)
        storage[1] = -14.5 * sin(14.5 * x[1] - 0.3) + 2. * x[1] + 0.2 + x[2]
        storage[2] = -14.5 * sin(14.5 * x[2] - 0.3) + 2. * x[2] + 0.2 + x[1]
    end

    x0 = [1., 1.]
    opt = (initial_guess)->optimize(_test_func2d_nograd, initial_guess, LBFGS())
    # minimum expected at ~[-0.195, -0.1 ≈ 0.0]
    ret = basinhopping(opt, x0, BasinhoppingParams(niter=200) )

    @test minimum(ret) < 1e-12
    @test isapprox(Optim.minimizer(ret), [-0.195, -0.1], atol=1e-4)

    x0 = [1., 1.]
    opt = (initial_guess)->optimize(_test_func2d, _test_func2d_grad!, initial_guess, LBFGS())
    # minimum expected at func([-0.19415263, -0.19415263]) = 0
    ret = basinhopping(opt, x0, BasinhoppingParams(niter=200))
    
    @test minimum(ret) < 1e-12
    @test isapprox(Optim.minimizer(ret), [-0.19415263, -0.19415263], atol=1e-5)
end
