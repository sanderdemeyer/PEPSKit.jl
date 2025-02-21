
using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using Zygote
using ChainrulesCore

## Setup

function three_dimensional_classical_ising(beta, J=1.0)
    K = beta * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4 -5 -6] :=
        M[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* q
    @tensor e_x[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * e[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_y[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * e[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_z[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * e[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    e = e_x + e_y + e_z

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end

## Test

# initialize
beta = 0.224 # slightly lower temperature than βc ≈ 0.2216544
O, M, E = three_dimensional_classical_ising(beta)
χenv = ℂ^12

# cover all different flavors
ctm_styles = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs = [HalfInfiniteProjector, FullInfiniteProjector]

@testset "PEPO CTMRG runthroughs for unitcell=$(unitcell)" for unitcell in
                                                               [(1, 1, 1), (1, 1, 2)]
    Random.seed!(81812781143)

    # contract
    T = InfinitePEPO(O; unitcell=unitcell)
    psi0 = initializePEPS(T, ComplexSpace(2))
    n = InfiniteSquareNetwork(psi, T)
    env0 = CTMRGEnv(n, χenv)

    @testset "PEPO CTMRG contraction using $ctm_style with $projector_alg" for (
        ctm_style, projector_alg
    ) in Iterators.product(
        ctm_styles, projector_algs
    )
        ctm_alg = ctm_style(; maxiter=150, projector_alg)
        env, = leading_boundary(env0, n, ctm_alg)
    end
end

@testset "Fixed-point computation for 3D classical ising model" begin
    Random.seed!(81812781143)

    # prep
    ctm_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=2)
    alg_rrule = LinSolver(;
        solver=KrylovKit.GMRES(; maxiter=30, tol=1e-6), iterscheme=:fixed
    )
    opt_alg = LBFGS(32; maxiter=20, gradtol=1e-4, verbosity=3)
    function pepo_retract(x, η, α)
        peps = deepcopy(x[1])
        peps.A .+= η.A .* α
        env2 = deepcopy(x[2])
        env3 = deepcopy(x[3])
        return (peps, env2, env3), η
    end

    # contract
    T = InfinitePEPO(O; unitcell=(1, 1, 1))
    psi0 = initializePEPS(T, ComplexSpace(2))
    env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), χenv)
    env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), χenv)

    # optimize free energy per site
    (psi_final, env2_final, env3_final), E, = optimize(
        (psi0, env2_0, env3_0), opt_alg; retract=pepo_retract, inner=PEPSKit.real_inner
    ) do (psi, env2, env3)
        E, gs = withgradient(psi) do ψ
            env2′, info = hook_pullback(leading_boundary, env2, n2, ctm_alg; alg_rrule)
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = hook_pullback(leading_boundary, env3, n3, ctm_alg; alg_rrule)
            ignore_derivatives() do
                update!(env2, env2′)
                update!(env3, env3′)
            end
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            return -log(real(λ3 / λ2))
        end
        g = only(gs)
        return E, g
    end

    # check energy
    n3_final = InfiniteSquareNetwork(psi_final, T)
    e = PEPSKit.contract_local_tensor((1, 1, 1), E, n3_final, env3_final)
    nrm3 = PEPSKit._contract_site((1, 1), n3_final, env3_final)

    e_per_link = e / nrm3 / 3

    @test e_per_link ≈ -0.53, atol = 1e-2

    # TODO: figure out what we should actually get
    # result does not seem to match the one from https://iopscience.iop.org/article/10.1088/0305-4470/31/29/007/pdf
    # beta = 0.2240, E ≈ 0.378615(26)
end
