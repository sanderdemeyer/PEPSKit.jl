using PEPSKit
using TensorKit
using KrylovKit
using JLD2


file = jldopen("examples/Ising_PEPO_p_2.jld2", "r")
O = file["O"]
close(file)

@tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]

pspace = ℂ^2
T = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
test = InfinitePEPS(T)

psi = InfinitePartitionFunction(Z)

χenv = 6
envtest = CTMRGEnv(test, ComplexSpace(χenv));
env0 = CTMRGEnv(psi, ComplexSpace(χenv));

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

env_init = leading_boundary(env0, psi, ctm_alg);
