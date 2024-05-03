var documenterSearchIndex = {"docs":
[{"location":"lib/lib/#Library","page":"Library","title":"Library","text":"","category":"section"},{"location":"lib/lib/","page":"Library","title":"Library","text":"Modules = [PEPSKit, PEPSKit.Defaults]","category":"page"},{"location":"lib/lib/#PEPSKit.AbstractPEPO","page":"Library","title":"PEPSKit.AbstractPEPO","text":"abstract type AbstractPEPO end\n\nAbstract supertype for a 2D projected entangled-pair operator.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.AbstractPEPS","page":"Library","title":"PEPSKit.AbstractPEPS","text":"abstract type AbstractPEPS end\n\nAbstract supertype for a 2D projected entangled-pair state.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.CTMRG","page":"Library","title":"PEPSKit.CTMRG","text":"struct CTMRG(; trscheme = TensorKit.notrunc(), tol = Defaults.ctmrg_tol,\n             maxiter = Defaults.ctmrg_maxiter, miniter = Defaults.ctmrg_miniter,\n             verbosity = 0, fixedspace = false)\n\nAlgorithm struct that represents the CTMRG algorithm for contracting infinite PEPS. The projector bond dimensions are set via trscheme which controls the truncation properties inside of TensorKit.tsvd. Each CTMRG run is converged up to tol where the singular value convergence of the corners as well as the norm is checked. The maximal and minimal number of CTMRG iterations is set with maxiter and miniter. Different levels of output information are printed depending on verbosity (0, 1 or 2). Regardless of the truncation scheme, the space can be kept fixed with fixedspace.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.CTMRGEnv","page":"Library","title":"PEPSKit.CTMRGEnv","text":"struct CTMRGEnv{C,T}\n\nCorner transfer-matrix environment containing unit-cell arrays of corner and edge tensors.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.CTMRGEnv-Union{Tuple{InfinitePEPS{P}}, Tuple{P}} where P","page":"Library","title":"PEPSKit.CTMRGEnv","text":"CTMRGEnv(peps::InfinitePEPS{P}; Venv=oneunit(spacetype(P)))\n\nCreate a random CTMRG environment from a PEPS tensor. The environment bond dimension defaults to one and can be specified using the Venv space.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.GeomSum","page":"Library","title":"PEPSKit.GeomSum","text":"struct GeomSum <: GradMode\n\nGradient mode for CTMRG using explicit evaluation of the geometric sum.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.InfinitePEPO","page":"Library","title":"PEPSKit.InfinitePEPO","text":"struct InfinitePEPO{T<:PEPOTensor}\n\nRepresents an infinte projected entangled-pair operator (PEPO) on a 3D cubic lattice.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Tuple{Integer, Integer, Integer}","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(d, D, L)\nInfinitePEPO(d, D, (Lx, Ly, Lz)))\n\nAllow users to pass in integers and specify unit cell.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Tuple{Integer, Integer}","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(d, D)\n\nAllow users to pass in integers.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Tuple{T} where T<:(TensorKit.AbstractTensorMap{S, 2, 4} where S<:TensorKit.ElementarySpace)","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(A)\n\nAllow users to pass in single tensor.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Union{Tuple{AbstractArray{T, 3}}, Tuple{T}} where T<:(TensorKit.AbstractTensorMap{S, 2, 4} where S<:TensorKit.ElementarySpace)","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(A::AbstractArray{T, 2})\n\nAllow users to pass in an array of tensors.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Union{Tuple{S}, Tuple{AbstractArray{S, 3}, AbstractArray{S, 3}}, Tuple{AbstractArray{S, 3}, AbstractArray{S, 3}, AbstractArray{S, 3}}} where S<:TensorKit.ElementarySpace","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(Pspaces, Nspaces, Espaces)\n\nAllow users to pass in arrays of spaces.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Union{Tuple{S}, Tuple{AbstractMatrix{S}, AbstractMatrix{S}}, Tuple{AbstractMatrix{S}, AbstractMatrix{S}, AbstractMatrix{S}}} where S<:TensorKit.ElementarySpace","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(Pspaces, Nspaces, Espaces)\n\nAllow users to pass in arrays of spaces, single layer special case.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPO-Union{Tuple{S}, Tuple{S, S}, Tuple{S, S, S}} where S<:TensorKit.ElementarySpace","page":"Library","title":"PEPSKit.InfinitePEPO","text":"InfinitePEPO(Pspace, Nspace, Espace)\n\nAllow users to pass in single space.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPS","page":"Library","title":"PEPSKit.InfinitePEPS","text":"struct InfinitePEPS{T<:PEPSTensor}\n\nRepresents an infinite projected entangled-pair state on a 2D square lattice.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.InfinitePEPS-Tuple{T} where T<:(TensorKit.AbstractTensorMap{S, 1, 4} where S<:TensorKit.ElementarySpace)","page":"Library","title":"PEPSKit.InfinitePEPS","text":"InfinitePEPS(A; unitcell=(1, 1))\n\nCreate an InfinitePEPS by specifying a tensor and unit cell.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPS-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:(TensorKit.AbstractTensorMap{S, 1, 4} where S<:TensorKit.ElementarySpace)","page":"Library","title":"PEPSKit.InfinitePEPS","text":"InfinitePEPS(A::AbstractArray{T, 2})\n\nAllow users to pass in an array of tensors.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPS-Union{Tuple{A}, Tuple{A, A, A}} where A<:(AbstractMatrix{<:Union{Int64, TensorKit.ElementarySpace}})","page":"Library","title":"PEPSKit.InfinitePEPS","text":"InfinitePEPS(f=randn, T=ComplexF64, Pspaces, Nspaces, Espaces)\n\nAllow users to pass in arrays of spaces.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.InfinitePEPS-Union{Tuple{S}, Tuple{S, S}, Tuple{S, S, S}} where S<:Union{Int64, TensorKit.ElementarySpace}","page":"Library","title":"PEPSKit.InfinitePEPS","text":"InfinitePEPS(f=randn, T=ComplexF64, Pspace, Nspace, [Espace]; unitcell=(1,1))\n\nCreate an InfinitePEPS by specifying its spaces and unit cell. Spaces can be specified either via Int or via ElementarySpace.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.ManualIter","page":"Library","title":"PEPSKit.ManualIter","text":"struct ManualIter <: GradMode\n\nGradient mode for CTMRG using manual iteration to solve the linear problem.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.NLocalOperator","page":"Library","title":"PEPSKit.NLocalOperator","text":"struct NLocalOperator{I<:AbstractInteraction}\n\nOperator in form of a AbstractTensorMap which is parametrized by an interaction type. Mostly, this is used to define Hamiltonian terms and observables.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.NaiveAD","page":"Library","title":"PEPSKit.NaiveAD","text":"struct NaiveAD <: GradMode\n\nGradient mode for CTMRG using AD.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.NearestNeighbor","page":"Library","title":"PEPSKit.NearestNeighbor","text":"struct NearestNeighbor <: AbstractInteraction\n\nInteraction representing nearest neighbor terms that act on two adjacent sites.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.OnSite","page":"Library","title":"PEPSKit.OnSite","text":"struct OnSite <: AbstractInteraction\n\nTrivial interaction representing terms that act on one isolated site.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPOTensor","page":"Library","title":"PEPSKit.PEPOTensor","text":"const PEPOTensor{S}\n\nDefault type for PEPO tensors with a single incoming and outgoing physical index, and 4 virtual indices, conventionally ordered as: O : P ⊗ P' ← N ⊗ E ⊗ S ⊗ W.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPO_∂∂AC","page":"Library","title":"PEPSKit.PEPO_∂∂AC","text":"struct PEPO_∂∂AC{T,O,P}\n\nRepresents the effective Hamiltonian for the one-site derivative of an MPS.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPO_∂∂C","page":"Library","title":"PEPSKit.PEPO_∂∂C","text":"struct PEPO_∂∂C{T<:GenericMPSTensor{S,N}}\n\nRepresents the effective Hamiltonian for the zero-site derivative of an MPS.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPSOptimize","page":"Library","title":"PEPSKit.PEPSOptimize","text":"PEPSOptimize{G}(; boundary_alg = CTMRG(), optimizer::OptimKit.OptimizationAlgorithm = LBFGS()\n                reuse_env::Bool = true, gradient_alg::G, verbosity::Int = 0)\n\nAlgorithm struct that represent PEPS ground-state optimization using AD. Set the algorithm to contract the infinite PEPS in boundary_alg; currently only CTMRG is supported. The optimizer computes the gradient directions based on the CTMRG gradient and updates the PEPS parameters. In this optimization, the CTMRG runs can be started on the converged environments of the previous optimizer step by setting reuse_env to true. Otherwise a random environment is used at each step. The CTMRG gradient itself is computed using the gradient_alg algorithm. Different levels of output verbosity can be activated using verbosity (0, 1 or 2).\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPSTensor","page":"Library","title":"PEPSKit.PEPSTensor","text":"const PEPSTensor{S}\n\nDefault type for PEPS tensors with a single physical index, and 4 virtual indices, conventionally ordered as: T : P ← N ⊗ E ⊗ S ⊗ W.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPSTensor-Union{Tuple{S}, Tuple{T}, Tuple{Any, Type{T}, S, S}, Tuple{Any, Type{T}, S, S, S}, Tuple{Any, Type{T}, Vararg{S, 4}}, Tuple{Any, Type{T}, Vararg{S, 5}}} where {T, S<:TensorKit.ElementarySpace}","page":"Library","title":"PEPSKit.PEPSTensor","text":"PEPSTensor(f, ::Type{T}, Pspace::S, Nspace::S,\n           [Espace::S], [Sspace::S], [Wspace::S]) where {T,S<:ElementarySpace}\nPEPSTensor(f, ::Type{T}, Pspace::Int, Nspace::Int,\n           [Espace::Int], [Sspace::Int], [Wspace::Int]) where {T}\n\nConstruct a PEPS tensor based on the physical, north, east, west and south spaces. Alternatively, only the space dimensions can be provided and ℂ is assumed as the field. The tensor elements are generated based on f and the element type is specified in T.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.PEPS_∂∂AC","page":"Library","title":"PEPSKit.PEPS_∂∂AC","text":"struct PEPS_∂∂AC{T,O1,O2}\n\nRepresents the effective Hamiltonian for the one-site derivative of an MPS.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#PEPSKit.PEPS_∂∂C","page":"Library","title":"PEPSKit.PEPS_∂∂C","text":"struct PEPS_∂∂C{T<:GenericMPSTensor{S,N}}\n\nRepresents the effective Hamiltonian for the zero-site derivative of an MPS.\n\n\n\n\n\n","category":"type"},{"location":"lib/lib/#MPSKit.expectation_value","page":"Library","title":"MPSKit.expectation_value","text":"MPSKit.expectation_value(peps::InfinitePEPS, env, O::NLocalOperator)\n\nEvaluate the expectation value of any NLocalOperator on each unit-cell entry of peps and env.\n\n\n\n\n\n","category":"function"},{"location":"lib/lib/#MPSKit.leading_boundary","page":"Library","title":"MPSKit.leading_boundary","text":"MPSKit.leading_boundary(state, alg::CTMRG, [envinit])\n\nContract state using CTMRG and return the CTM environment. Per default, a random initial environment is used.\n\n\n\n\n\n","category":"function"},{"location":"lib/lib/#PEPSKit._rrule-Tuple{Nothing, ChainRulesCore.RuleConfig, Any, Vararg{Any}}","page":"Library","title":"PEPSKit._rrule","text":"_rrule(alg_rrule, config, f, args...; kwargs...) -> ∂f, ∂args...\n\nCustomize the pullback of a function f. This function can specialize on its first argument in order to have multiple implementations for a pullback. If no specialization is needed, the default alg_rrule=nothing results in the default AD pullback.\n\nwarning: Warning\nNo tangent is expected for the alg_rrule argument\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.check_elementwise_convergence-Tuple{CTMRGEnv, CTMRGEnv}","page":"Library","title":"PEPSKit.check_elementwise_convergence","text":"check_elementwise_convergence(envfinal, envfix; atol=1e-6)\n\nCheck if the element-wise difference of the corner and edge tensors of the final and fixed CTMRG environments are below some tolerance.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.costfun-Tuple{InfinitePEPS, Any, NLocalOperator{NearestNeighbor}}","page":"Library","title":"PEPSKit.costfun","text":"costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})\n\nCompute the expectation value of a nearest-neighbor operator. This is used to evaluate and differentiate the energy in ground-state PEPS optimizations.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.ctmrg_iter-Union{Tuple{T}, Tuple{C}, Tuple{Any, CTMRGEnv{C, T}, CTMRG}} where {C, T}","page":"Library","title":"PEPSKit.ctmrg_iter","text":"ctmrg_iter(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}\n\nPerform one iteration of CTMRG that maps the state and env to a new environment, and also return the truncation error. One CTMRG iteration consists of four left_move calls and 90 degree rotations, such that the environment is grown and renormalized in all four directions.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.fixedpoint-Union{Tuple{T}, Tuple{InfinitePEPS{T}, Any, PEPSOptimize}, Tuple{InfinitePEPS{T}, Any, PEPSOptimize, CTMRGEnv}} where T","page":"Library","title":"PEPSKit.fixedpoint","text":"fixedpoint(ψ₀::InfinitePEPS{T}, H, alg::PEPSOptimize, [env₀::CTMRGEnv]) where {T}\n\nOptimize ψ₀ with respect to the Hamiltonian H according to the parameters supplied in alg. The initial environment env₀ serves as an initial guess for the first CTMRG run. By default, a random initial environment is used.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.fpgrad","page":"Library","title":"PEPSKit.fpgrad","text":"fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y0, alg)\n\nCompute the gradient of the cost function for CTMRG by solving the following equation:\n\ndx = ∑ₙ (∂f∂x)ⁿ ∂f∂A dA = (1 - ∂f∂x)⁻¹ ∂f∂A dA\n\nwhere ∂F∂x is the gradient of the cost function with respect to the PEPS tensors, ∂f∂x is the partial gradient of the CTMRG iteration with respect to the environment tensors, ∂f∂A is the partial gradient of the CTMRG iteration with respect to the PEPS tensors, and y0 is the initial guess for the fixed-point iteration. The function returns the gradient dx of the fixed-point iteration.\n\n\n\n\n\n","category":"function"},{"location":"lib/lib/#PEPSKit.gauge_fix-Union{Tuple{T}, Tuple{C}, Tuple{CTMRGEnv{C, T}, CTMRGEnv{C, T}}} where {C, T}","page":"Library","title":"PEPSKit.gauge_fix","text":"gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}\n\nFix the gauge of envfinal based on the previous environment envprev. This assumes that the envfinal is the result of one CTMRG iteration on envprev. Given that the CTMRG run is converged, the returned environment will be element-wise converged to envprev.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.hook_pullback-Tuple{Any, Vararg{Any}}","page":"Library","title":"PEPSKit.hook_pullback","text":"hook_pullback(f, args...; alg_rrule=nothing, kwargs...)\n\nWrapper function to customize the pullback of a function f. This function is equivalent to f(args...; kwargs...), but the pullback can be customized by implementing the following function:\n\n_rrule(alg_rrule, config, f, args...; kwargs...) -> NoTangent(), ∂f, ∂args...\n\nThis function can specialize on its first argument in order to customize the pullback. If no specialization is needed, the default alg_rrule=nothing results in the default AD pullback.\n\nSee also _rrule.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.left_move-Union{Tuple{T}, Tuple{C}, Tuple{Any, CTMRGEnv{C, T}, CTMRG}} where {C, T}","page":"Library","title":"PEPSKit.left_move","text":"left_move(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}\n\nGrow, project and renormalize the environment env in west direction. Return the updated environment as well as the projectors and truncation error.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.operator_env","page":"Library","title":"PEPSKit.operator_env","text":"operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::AbstractInteraction)\n\nContract a PEPS and a CTMRG environment to form an operator environment. The open bonds correspond to the indices of an operator with the specified AbstractInteraction type.\n\n\n\n\n\n","category":"function"},{"location":"lib/lib/#PEPSKit.projector_type-Tuple{DataType, Any}","page":"Library","title":"PEPSKit.projector_type","text":"projector_type(T::DataType, size)\n\nCreate two arrays of specified size that contain undefined tensors representing left and right acting projectors, respectively. The projector types are inferred from the TensorMap type T which avoids having to recompute transpose tensors.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.rotate_north-Tuple{Any, Any}","page":"Library","title":"PEPSKit.rotate_north","text":"rotate_north(t, dir)\n\nRotate north direction of t to dir by successive applications of rotl90.\n\n\n\n\n\n","category":"method"},{"location":"lib/lib/#PEPSKit.@diffset-Tuple{Any}","page":"Library","title":"PEPSKit.@diffset","text":"@diffset assign\n\nHelper macro which allows in-place operations in the forward-pass of Zygote, but resorts to non-mutating operations in the backwards-pass. The expression assign should assign an object to an pre-existing AbstractArray and the use of updating operators is also possible. This is especially needed when in-place assigning tensors to unit-cell arrays of environments.\n\n\n\n\n\n","category":"macro"},{"location":"lib/lib/#PEPSKit.@showtypeofgrad-Tuple{Any}","page":"Library","title":"PEPSKit.@showtypeofgrad","text":"@showtypeofgrad(x)\n\nMacro utility to show to type of the gradient that is about to accumulate for x.\n\nSee also Zygote.@showgrad.\n\n\n\n\n\n","category":"macro"},{"location":"lib/lib/#PEPSKit.Defaults","page":"Library","title":"PEPSKit.Defaults","text":"module Defaults\n    const ctmrg_maxiter = 100\n    const ctmrg_miniter = 4\n    const ctmrg_tol = 1e-12\n    const fpgrad_maxiter = 100\n    const fpgrad_tol = 1e-6\nend\n\nModule containing default values that represent typical algorithm parameters.\n\nctmrg_maxiter = 100: Maximal number of CTMRG iterations per run\nctmrg_miniter = 4: Minimal number of CTMRG carried out\nctmrg_tol = 1e-12: Tolerance checking singular value and norm convergence\nfpgrad_maxiter = 100: Maximal number of iterations for computing the CTMRG fixed-point gradient\nfpgrad_tol = 1e-6: Convergence tolerance for the fixed-point gradient iteration\n\n\n\n\n\n","category":"module"},{"location":"man/intro/","page":"Manual","title":"Manual","text":"Coming soon.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"For now, refer to the examples folder on GitHub.","category":"page"},{"location":"#PEPSKit.jl","page":"Home","title":"PEPSKit.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tools for working with projected entangled-pair states","category":"page"},{"location":"","page":"Home","title":"Home","text":"It contracts, it optimizes, it may be broken at any point.","category":"page"}]
}
