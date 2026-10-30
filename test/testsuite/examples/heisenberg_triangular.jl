using Test
using Random
using PEPSKit
using TensorKit
using Adapt

function examples_heisenberg_triangular(AT)
    return @testset "Triangular AF Heisenberg ($AT)" begin
        # References
        # ----------
        # Variational optimization of projected entangled-pair states on the triangular lattice
        # J. Naumann, J. Eisert, and P. Schmoll
        # Phys. Rev. B 113, 045117 (2026), arXiv:2510.04907
        # Table 1, native triangular CTMRG with a single-site spiral (120°) ansatz at D = 2
        E_ref = -0.54419

        # initialize parameters
        Dbond = 2
        χenv = 16
        Pspace = ComplexSpace(2)
        Vspace = ComplexSpace(Dbond)
        boundary_alg = (; alg = :SimultaneousCTMRGTriangular, tol = 1.0e-9, maxiter = 300, miniter = 4)
        gradient_alg = (; solver_alg = (; alg = :GMRES, krylovdim = 30), tol = 1.0e-7, maxiter = 10)
        optimizer_alg = (; tol = 1.0e-4, maxiter = 100)
        heisenberg(lattice) = adapt(AT, heisenberg_XYZ(lattice; Jx = 1.0, Jy = 1.0, Jz = 1.0))

        function optimize_energy_per_site(H, peps₀, nsites; symmetrization = nothing)
            env₀, = leading_boundary(
                CTMRGEnvTriangular(randn, ComplexF64, Vspace ⊗ Vspace', ComplexSpace(χenv); unitcell = size(peps₀)),
                peps₀; boundary_alg...,
            )
            _, _, E, = fixedpoint(
                H, peps₀, env₀;
                boundary_alg, gradient_alg, optimizer_alg, symmetrization, precondition_alg = nothing,
            )
            return E / nsites
        end

        # the 120° order fits the three-sublattice unit cell
        Random.seed!(1)
        symmetrization = TriangularThreeSublattice()
        peps₀ = symmetrize!(
            adapt(AT, InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace; unitcell = (3, 3))),
            symmetrization,
        )
        e_three = optimize_energy_per_site(
            heisenberg(InfiniteTriangularThreeSublattice()), peps₀, 3; symmetrization
        )
        @info "Three-sublattice energy per site = $e_three"
        @test e_three ≈ E_ref atol = 3.0e-3

        # a 2 × 2 unit cell frustrates the 120° order
        Random.seed!(1)
        peps₀ = adapt(AT, InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace; unitcell = (2, 2)))
        e_2x2 = optimize_energy_per_site(heisenberg(InfiniteTriangular(2, 2)), peps₀, 4)
        @info "2 × 2 energy per site = $e_2x2"
        @test e_three < e_2x2
    end
end
