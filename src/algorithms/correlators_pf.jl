function correlator_horizontal(
    network::InfinitePartitionFunction,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,2,3},
    O₂::AbstractTensorMap{T,S,3,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}; 
    ) where {T,S}
    
    corr = T[]
    (Nr, Nc) = size(network)
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])
    @assert r == r₂ "Not a horizontal correlation function."
    @assert c₁ < c₂ "The first column index must be less than the second."

    @autoopt @tensor left_side[χS DE Dstring; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DN; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DW; χ3] * O₁[DW DS; DN DE Dstring] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DS; χ1]
    @autoopt @tensor left_side_norm[χS DE; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DN; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DW; χ3] * network[mod1(r, Nr), mod1(c₁, Nc)][DW DS; DN DE] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DS; χ1]
    for c = c₁+1:c₂
        final = @autoopt @tensor left_side[χ6 DW Dstring; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DN; χ2] * env.corners[1, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        O₂[Dstring DW DS; DN DE] * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DE; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DS; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        final_norm = @autoopt @tensor left_side_norm[χ6 DW; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DN; χ2] * env.corners[1, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] *  env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DE; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DS; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        push!(corr, final / final_norm)
        if c ≠ c₂
            @autoopt @tensor left_side[χS DE Dstring; χN] = left_side[χ1 DW Dstring; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DN; χN] * 
            network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] *
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DS; χ1]
            @autoopt @tensor left_side_norm[χS DE; χN] = left_side_norm[χ1 DW; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DN; χN] * 
            network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] * 
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DS; χ1]
        end
    end
    return corr
end

function correlator_horizontal(
    network::InfinitePartitionFunction,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,2,2},
    O₂::AbstractTensorMap{T,S,2,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}; 
    ) where {T,S}
    
    corr = T[]
    (Nr, Nc) = size(network)
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])
    @assert r == r₂ "Not a horizontal correlation function."
    @assert c₁ < c₂ "The first column index must be less than the second."

    @autoopt @tensor left_side[χS DE; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DN; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DW; χ3] * O₁[DW DS; DN DE] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DS; χ1]
    @autoopt @tensor left_side_norm[χS DE; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DN; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DW; χ3] * network[mod1(r, Nr), mod1(c₁, Nc)][DW DS; DN DE] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DS; χ1]
    for c = c₁+1:c₂
        final = @autoopt @tensor left_side[χ6 DW; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DN; χ2] * env.corners[1, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        O₂[DW DS; DN DE] * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DE; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DS; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        final_norm = @autoopt @tensor left_side_norm[χ6 DW; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DN; χ2] * env.corners[1, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] *  env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DE; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DS; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        push!(corr, final / final_norm)
        if c ≠ c₂
            @autoopt @tensor left_side[χS DE; χN] = left_side[χ1 DW; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DN; χN] * 
            network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] *
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DS; χ1]
            @autoopt @tensor left_side_norm[χS DE; χN] = left_side_norm[χ1 DW; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DN; χN] * 
            network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] * 
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DS; χ1]
        end
    end
    return corr
end

function correlator_horizontal(
    ρ::InfinitePEPO,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,1,1},
    O₂::AbstractTensorMap{T,S,1,1},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}
    ) where {T,S}
    
    network = trace_out(ρ)

    (Nr, Nc, h) = size(ρ)
    @assert h == 1 "Only one-layer PEPOs are supported"
    (r₁, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])

    @tensor Oleft[-4 -3; -1 -2] := twist(ρ[mod1(r₁, Nr), mod1(c₁, Nc)], 2)[1 2; -1 -2 -3 -4] * O₁[2; 1]
    @tensor Oright[-4 -3; -1 -2] := twist(ρ[mod1(r₂, Nr), mod1(c₂, Nc)], 2)[1 2; -1 -2 -3 -4] * O₂[2; 1]

    return correlator_horizontal(network, env, Oleft, Oright, inds)
end

function correlator_horizontal(
    ρ::InfinitePEPO,
    env::CTMRGEnv,
    O::AbstractTensorMap{T,S,2,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}
    ) where {T,S}
    
    network = trace_out(ρ)

    (Nr, Nc, h) = size(ρ)
    @assert h == 1 "Only one-layer PEPOs are supported"
    (r₁, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])

    @tensor t[-1 -2 -3 -4; -5 -6 -7 -8] := twist(ρ[mod1(r₁, Nr), mod1(c₁, Nc)], 2)[1 2; -1 -2 -3 -4] * twist(ρ[mod1(r₂, Nr), mod1(c₂, Nc)], 2)[3 4; -5 -6 -7 -8] * O[2 4; 1 3]
    U, Σ, V = tsvd(t)
    O₁ = permute(U * sqrt(Σ), ((4,3),(1,2,5)))
    O₂ = permute(sqrt(Σ) * V, ((1,5,4),(2,3)))

    return correlator_horizontal(network, env, O₁, O₂, inds)
end
