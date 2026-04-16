using Test
using Random
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using Zygote
using LinearAlgebra
using KrylovKit

using PEPSKit: unitcell, norm_preserving_retract

sd = 123456
Random.seed!(sd)

D = 2
χ = 7
T = ComplexF64

ctmrg_tol = 1.0e-8
ctmrg_maxiter = 300
ctmrg_verbosity = 2

const ising_βc_triangular = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(1.0) / BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_triangular::BigFloat = -3.20253248660790791834355252025862951439

function classical_ising_triangular(β)
    t = Float64[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3; -4 -5 -6] := O[1 2 3; 4 5 6] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4] * nt[-5; 5] * nt[-6; 6]
    @tensor o2[-1 -2 -3; -4 -5 -6] := o[1 2 3; 4 5 6] * H[-1; 1] * H[-2; 2] * H[-3; 3] * H[-4; 4] * H[-5; 5] * H[-6; 6]
    return TensorMap(o2, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
end

@testset "CTM_triangular - Random tensor" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(79413165445)
            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            pspace = ComplexSpace(2)
            vspace = ComplexSpace(D)
            envspace = ComplexSpace(χ)

            ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)
            pf = randn(T, vspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ vspace)
            sandwiches = [pf, (ket, bra)]
            # sandwiches = [(ket, bra), pf]
            unitcell = (2, 2)

            for (sandwich, V) in zip(sandwiches, [vspace, vspace ⊗ vspace'])
                # for (sandwich, vspace) in zip(sandwiches, [vspace ⊗ vspace', vspace])
                network = InfiniteTriangularNetwork(fill(sandwich, unitcell))
                env₀ = CTMRGEnvTriangular(randn, T, V, envspace; unitcell)
                env, info = leading_boundary(env₀, network, alg)
                @test info.convergence_metric < 1.0 # this is not much of a test
            end
        end
    end
end

@testset "CTM_triangular - Classical Ising" begin
    χ_local = 20
    T_local = Float64

    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(156484561351)

            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            sz = (1, 1)
            T = classical_ising_triangular(ising_βc_triangular)
            pf = InfiniteTriangularNetwork(fill(T, sz))

            vspace = codomain(T)[1]
            envspace = ComplexSpace(χ_local)
            env₀ = CTMRGEnvTriangular(randn, T_local, vspace, envspace; unitcell = sz)
            env, info = leading_boundary(env₀, pf, alg)

            nw_value = network_value(pf, env)
            lz = real(log(nw_value))
            fs = lz * -1 / ising_βc_triangular
            @test fs ≈ f_onsager_triangular rtol = 1.0e-4
        end
    end
end
