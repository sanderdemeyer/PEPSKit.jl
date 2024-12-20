using PEPSKit
using TensorKit
using JLD2

file = jldopen("Ising_PEPO_p_2.jld2", "r")
O = file["O"]
close(file)

@tensor Z[-1 -2; -3 -4] = O[1 1; -1 -2 -3 -4]

χenv = 16
env0 = CTMRGEnv(Z, ComplexSpace(χenv));

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

env_init = leading_boundary(env0, psi_init, ctm_alg);
