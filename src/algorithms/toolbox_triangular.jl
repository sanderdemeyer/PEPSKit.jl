function network_value(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular)
    return prod(Iterators.product(axes(network)...)) do (r, c)
        nw_corners = complex(_contract_corners((r, c), network, env))
        nw_full = complex(_contract_site_large((r, c), network, env))
        nw_0 = complex(_contract_edges_0((r, c), network, env))
        nw_60 = complex(_contract_edges_60((r, c), network, env))
        nw_120 = complex(_contract_edges_120((r, c), network, env))
        return (nw_full * nw_corners^2 / (nw_0 * nw_60 * nw_120))^(1 / 3)
    end
end

function energy(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, onesite_op, twosite_op) where {P <: PEPSSandwichTriangular}
    E = sum(Iterators.product(axes(network)...)) do (r, c)
        expval_onesite = _contract_corners((r, c), network, env, onesite_op) / _contract_corners((r, c), network, env)
        expval_twosite = _contract_edges_0((r, c), network, env, twosite_op) / _contract_edges_0((r, c), network, env)
        return expval_onesite + expval_twosite
    end
    ignore_derivatives() do
        isapprox(imag(E), 0; atol = sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end
function energy(state::InfinitePEPSTriangular, env::CTMRGEnvTriangular, args...)
    return energy(InfiniteTriangularNetwork(state), env, args...)
end

# TODO: write a proper cost_function that works on an InfinitePEPSTriangular and a LocalOperator, and use it in the optimization loop. For now, we just compute the energy using the network contraction functions above.
