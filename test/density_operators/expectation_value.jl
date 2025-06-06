using TensorKit, PEPSKit
using Test
using Random
using JLD2

Random.seed!(18789479)

T = ComplexF64

pspace = ℂ^2
vspace = ℂ^2
envspace = ℂ^12
intspace = ℂ^3

ket = InfinitePEPS(randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace'); unitcell = (2,2))
env, = leading_boundary(CTMRGEnv(ket, envspace), ket, SimultaneousCTMRG())

O = randn(T, pspace ⊗ pspace, pspace ⊗ pspace)
O1 = randn(T, pspace, pspace)
left = CartesianIndex(1,1)
right = CartesianIndex(1,5)
corr = correlator_horizontal(ket, ket, env, O, (left, right))
corr2 = correlator_horizontal(ket, ket, env, O1, O1, (left, right))


pf = InfinitePartitionFunction(randn(T, vspace ⊗ vspace, vspace ⊗ vspace))
env, = leading_boundary(CTMRGEnv(pf, envspace), pf, SimultaneousCTMRG())
O₁ = randn(T, vspace ⊗ vspace, vspace ⊗ vspace ⊗ intspace)
O₂ = randn(T, intspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace)
corr3 = correlator_horizontal(pf, env, O₁, O₂, (left, right))

ρ = InfinitePEPO(randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace'), unitcell = (2, 2, 1))
env, = leading_boundary(CTMRGEnv(InfiniteSquareNetwork(ρ), envspace), InfiniteSquareNetwork(ρ), SimultaneousCTMRG());
O_onesite = randn(T, pspace, pspace)
O_twosite = randn(T, pspace ⊗ pspace, pspace ⊗ pspace)


corr4 = correlator_horizontal(ρ, env, O_twosite, (left, right))
corr5 = correlator_horizontal(ρ, env, O_onesite, O_onesite, (left, right))

@testset "Test expectation values of InfinitePartitionFunction" begin
    pspace = ℂ^2
    vspace = ℂ^2
    intspace = ℂ^1
    envspace = ℂ^12

    pftensor = randn(T, vspace ⊗ vspace, vspace ⊗ vspace)
    PF = InfinitePartitionFunction(pftensor)
    env, = leading_boundary(CTMRGEnv(PF, envspace), PF, SimultaneousCTMRG(; maxiter = 100))
    O = copy(pftensor)
    OL = TensorMap(O.data, vspace ⊗ vspace, vspace ⊗ vspace ⊗ intspace)
    OR = TensorMap(O.data, intspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace)
    OLR = TensorMap(O.data, intspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ intspace)

    expval1 = expectation_value(PF, ([CartesianIndex(1,1) => O],), env)
    expval2 = expectation_value(PF, ([CartesianIndex(2,1) => OL, CartesianIndex(2,2) => OR],), env)
    expval3 = expectation_value(PF, ([CartesianIndex(1,1) => O],[CartesianIndex(2,1) => OL, CartesianIndex(2,2) => OR],), env)

    @test expval1 ≈ 1.0
    @test expval2 ≈ 1.0
    @test expval3 ≈ 1.0
end

@testset "Test expectation values of InfinitePEPO" begin
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    intspace = Vect[fℤ₂](0 => 2, 1 => 1)
    envspace = Vect[fℤ₂](0 => 5, 1 => 5)

    # O = InfinitePEPO(randn(pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace'))
    file = jldopen("ClusterExpansion_test_smaller.jld2")
    O = InfinitePEPO(file["O"])
    close(file)
    H = InfinitePEPO(permute(id(T, pspace ⊗ intspace ⊗ intspace), ((1,4),(5,6,2,3))))

    ctm_alg = SimultaneousCTMRG(; maxiter = 300)

    (Nr, Nc) = (2,2)
    lattice = InfiniteSquare(Nr, Nc)
    pspaces = fill(pspace, Nr, Nc)
    H_LO = PEPSKit.LocalOperator(pspaces, ((idx,) => id(pspace) for idx in PEPSKit.vertices(lattice))...,)

    expval_rho1 = expectation_value(O, H, envspace, ctm_alg)
    expval_rho2 = expectation_value(O, H_LO, envspace, ctm_alg) / (Nr*Nc)
    println(expval_rho1, " ", expval_rho2)
    @test expval_rho1 ≈ 1.0 atol = 1e-3
    @test expval_rho2 ≈ 1.0
end