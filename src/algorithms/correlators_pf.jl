function correlator_horizontal(
    network::InfinitePartitionFunction,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,2,3},
    O₂::AbstractTensorMap{T,S,3,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}; 
    ) where {T,S}
    
    corr = T[]
    (Nr, Nc) = size(ket)
    (r, c₁) = Tuple(inds[1])
    (r2, c₂) = Tuple(inds[2])
    @assert r == r2 "Not a horizontal correlation function."

    @autoopt @tensor left_side[χS DE Dstring; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DNt DNb; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DWt DWb; χ3] * O₁[DW DS; DN DE Dstring] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DSt DSb; χ1]
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
            @autoopt @tensor left_side_norm[χS DEt DEb; χN] = left_side_norm[χ1 DWt DWb; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DN; χN] * 
            network[mod1(r, Nr), mod1(c, Nc)][DW DS; DN DE] * 
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DS; χ1]
        end
    end
    return corr
end
