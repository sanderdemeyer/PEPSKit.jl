abstract type CTMRGAlgorithmTriangular end

"""
    leading_boundary(env₀::CTMRGEnvTriangular, network, alg::CTMRGAlgorithmTriangular)

Return the contracted environment and convergence information. `info.converged` and
`info.convergence_error` describe convergence; `info.convergence_metric` is retained as
an alias for the latter. `info.contraction_metrics` contains `corner_truncation_error`
and `edge_truncation_error`, maximized over all sites and directions of the final
iteration, and their maximum `truncation_error`.

Errors are discarded norms divided by the retained SVD spectrum norm, following
PEPSKit's corner-projector convention. They include singular values suppressed by
the inverse-square-root cutoff, but exclude preliminary seed and conditioning SVDs.
These are local truncation diagnostics, not errors on the final observables.
"""
function leading_boundary(env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return leading_boundary(env₀, network, alg)
end
function leading_boundary(
        env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork, alg::CTMRGAlgorithmTriangular
    )
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))
    return with_verbosity(; alg.verbosity) do
        env = deepcopy(env₀)
        S_old = DiagonalTensorMap.(id.(domain.(env₀.C)))
        η = one(real(scalartype(network)))
        ctmrg_loginit!(log, η, network, env₀)
        local info_iter
        converged = false
        for iter in 1:(alg.maxiter)
            env, info_iter = ctmrg_iteration(network, env, alg)
            η = calc_convergence(info_iter.S, S_old)
            S_old = copy(info_iter.S)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, network, env)
                converged = true
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, network, env)
            else
                ctmrg_logiter!(log, iter, η, network, env)
            end
        end
        info = (;
            converged, convergence_error = η, convergence_metric = η,
            contraction_metrics = info_iter.contraction_metrics,
        )
        return env, info
    end
end
function leading_boundary(env₀::CTMRGEnvTriangular, state, args...; kwargs...)
    return leading_boundary(env₀, InfiniteTriangularNetwork(state), args...; kwargs...)
end

function calc_convergence(S_new::Array{T, 3}, S_old::Array{T, 3}) where {E, S, T <: DiagonalTensorMap{E, S}}
    ε = Inf
    function dist(S1, S2)
        if space(S1) == space(S2)
            return norm(S1^4 - S2^4)
        else
            return Inf
        end
    end
    return maximum(dist.(S_new, S_old))
end
