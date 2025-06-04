function correlator_horizontal(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,1,2},
    O₂::AbstractTensorMap{T,S,2,1},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}; 
    ) where {T,S}

    @assert size(ket) == size(bra) "The ket and bra must have the same unit cell."
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])
    @assert r == r₂ "Not a horizontal correlation function."
    @assert c₁ < c₂ "The first column index must be less than the second."

    (Nr, Nc) = size(ket)
    corr = T[]

    @autoopt @tensor left_side[χS DEt Dstring DEb; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DNt DNb; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DWt DWb; χ3] * ket[mod1(r, Nr), mod1(c₁, Nc)][dt; DNt DEt DSt DWt] * conj(bra[r,c₁][db; DNb DEb DSb DWb]) * O₁[db; dt Dstring] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DSt DSb; χ1]
    @autoopt @tensor left_side_norm[χS DEt DEb; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DNt DNb; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DWt DWb; χ3] * ket[mod1(r, Nr), mod1(c₁, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c₁, Nc)][d; DNb DEb DSb DWb]) * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DSt DSb; χ1]
    for c = c₁+1:c₂
        final = @autoopt @tensor left_side[χ6 DWt Dstring DWb; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DNt DNb; χ2] * env.corners[2, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        ket[mod1(r, Nr), mod1(c, Nc)][dt; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][db; DNb DEb DSb DWb]) * O₂[Dstring db; dt] * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DEt DEb; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DSt DSb; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        final_norm = @autoopt @tensor left_side_norm[χ6 DWt DWb; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DNt DNb; χ2] * env.corners[2, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DEt DEb; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DSt DSb; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]

        push!(corr, final / final_norm)
        if c ≠ c₂
            @autoopt @tensor left_side[χS DEt Dstring DEb; χN] = left_side[χ1 DWt Dstring DWb; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DNt DNb; χN] * 
            ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) *
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DSt DSb; χ1]
            @autoopt @tensor left_side_norm[χS DEt DEb; χN] = left_side_norm[χ1 DWt DWb; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DNt DNb; χN] * 
            ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) * 
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DSt DSb; χ1]
        end
    end
    return corr
end

function correlator_horizontal(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,1,1},
    O₂::AbstractTensorMap{T,S,1,1},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}
    ) where {T,S}

    corr = T[]
    (Nr, Nc) = size(ket)
    @assert size(ket) == size(bra) "The ket and bra must have the same unit cell."
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])
    @assert r == r₂ "Not a horizontal correlation function."
    @assert c₁ < c₂ "The first column index must be less than the second."

    @autoopt @tensor left_side[χS DEt DEb; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DNt DNb; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DWt DWb; χ3] * ket[mod1(r, Nr), mod1(c₁, Nc)][dt; DNt DEt DSt DWt] * conj(bra[r,c₁][db; DNb DEb DSb DWb]) * O₁[db; dt] * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DSt DSb; χ1]
    @autoopt @tensor left_side_norm[χS DEt DEb; χN] := env.corners[1,_prev(r, Nr), _prev(c₁, Nc)][χ3; χ4] * env.edges[1, _prev(r, Nr), mod1(c₁, Nc)][χ4 DNt DNb; χN] * 
    env.edges[4, mod1(r, Nr),  _prev(c₁, Nc)][χ2 DWt DWb; χ3] * ket[mod1(r, Nr), mod1(c₁, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c₁, Nc)][d; DNb DEb DSb DWb]) * 
    env.corners[4, _next(r, Nr), _prev(c₁, Nc)][χ1; χ2] * env.edges[3,_next(r, Nr), mod1(c₁, Nc)][χS DSt DSb; χ1]
    for c = c₁+1:c₂
        final = @autoopt @tensor left_side[χ6 DWt DWb; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DNt DNb; χ2] * env.corners[2, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        ket[mod1(r, Nr), mod1(c, Nc)][dt; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][db; DNb DEb DSb DWb]) * O₂[db; dt] * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DEt DEb; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DSt DSb; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        final_norm = @autoopt @tensor left_side_norm[χ6 DWt DWb; χ1] * 
        env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ1 DNt DNb; χ2] * env.corners[2, _prev(r, Nr), _next(c, Nc)][χ2; χ3] *
        ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) * env.edges[2, mod1(r, Nr), _next(c, Nc)][χ3 DEt DEb; χ4] * 
        env.edges[3, _next(r, Nr), mod1(c, Nc)][χ5 DSt DSb; χ6] * env.corners[3, _next(r, Nr), _next(c, Nc)][χ4; χ5]
        push!(corr, final / final_norm)
        if c ≠ c₂
            @autoopt @tensor left_side[χS DEt DEb; χN] = left_side[χ1 DWt DWb; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DNt DNb; χN] * 
            ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) *
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DSt DSb; χ1]
            @autoopt @tensor left_side_norm[χS DEt DEb; χN] = left_side_norm[χ1 DWt DWb; χ4] * env.edges[1, _prev(r, Nr), mod1(c, Nc)][χ4 DNt DNb; χN] * 
            ket[mod1(r, Nr), mod1(c, Nc)][d; DNt DEt DSt DWt] * conj(bra[mod1(r, Nr), mod1(c, Nc)][d; DNb DEb DSb DWb]) * 
            env.edges[3,_next(r, Nr), mod1(c, Nc)][χS DSt DSb; χ1]
        end
    end
    return corr
end

function correlator_horizontal(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O::AbstractTensorMap{T,S,2,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}}
    ) where {T,S}
    U, Σ, V = tsvd(O, ((1,3),(2,4)))
    O₁ = permute(U * sqrt(Σ), ((1,),(2,3)))
    O₂ = permute(sqrt(Σ) * V, ((1,2),(3,)))
    return correlator_horizontal(ket, bra, env, O₁, O₂, inds)
end

function correlator_horizontal(
    ket::PEPSTensor,
    bra::PEPSTensor,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,1,2},
    O₂::AbstractTensorMap{T,S,2,1},
    R::Int 
) where {T,S}
    @autoopt @tensor transfermatrix[χNW DWt DWb χSW; χNE DEt DEb χSE] := env.edges[1,1,1][χNW DNt DNb; χNE] * ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) * env.edges[3,1,1][χSE DSt DSb; χSW]
    U, Σ, V = tsvd(transfermatrix)
    
    @autoopt @tensor left[Dstring; Dsvd] := env.corners[1,1,1][χ3; χ4] * env.edges[1,1,1][χ4 DNt DNb; χN] * 
    env.edges[4,1,1][χ2 DWt DWb; χ3] * ket[dt; DNt DEt DSt DWt] * conj(bra[db; DNb DEb DSb DWb]) * O₁[db; dt Dstring] * 
    env.corners[4,1,1][χ1; χ2] * env.edges[3,1,1][χS DSt DSb; χ1] * U[χN DEt DEb χS; Dsvd]

    @autoopt @tensor left_norm[Dsvd] := env.corners[1,1,1][χ3; χ4] * env.edges[1,1,1][χ4 DNt DNb; χN] * 
    env.edges[4,1,1][χ2 DWt DWb; χ3] * ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) * 
    env.corners[4,1,1][χ1; χ2] * env.edges[3,1,1][χS DSt DSb; χ1] * U[χN DEt DEb χS; Dsvd]

    @autoopt @tensor right[Dstring; Dsvd] := V[Dsvd; χ1 DWt DWb χ6] * 
    env.edges[1,1,1][χ1 DNt DNb; χ2] * env.corners[2,1,1][χ2; χ3] *
    ket[dt; DNt DEt DSt DWt] * conj(bra[db; DNb DEb DSb DWb]) * O₂[Dstring db; dt] * env.edges[2,1,1][χ3 DEt DEb; χ4] * 
    env.edges[3,1,1][χ5 DSt DSb; χ6] * env.corners[3,1,1][χ4; χ5]

    @autoopt @tensor right_norm[Dsvd] := V[Dsvd; χ1 DWt DWb χ6] * 
    env.edges[1,1,1][χ1 DNt DNb; χ2] * env.corners[2,1,1][χ2; χ3] *
    ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) * env.edges[2,1,1][χ3 DEt DEb; χ4] * 
    env.edges[3,1,1][χ5 DSt DSb; χ6] * env.corners[3,1,1][χ4; χ5]

    @tensor t[-1; -2] := left[1; -1] * right[1; -2]
    @tensor t_norm[-1; -2] := left_norm[-1] * right_norm[-2]

    function corr(r)
        c = @tensor t[1; 2] * (D^(r-1))[1; 2]
        nrm = @tensor t_norm[1; 2] * (D^(r-1))[1; 2]
        return c / nrm
    end
    return corr.(1:R)
end

function correlator_horizontal(
    ket::PEPSTensor,
    bra::PEPSTensor,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S,1,1},
    O₂::AbstractTensorMap{T,S,1,1},
    R::Int 
) where {T,S}
    @autoopt @tensor transfermatrix[χNW DWt DWb χSW; χNE DEt DEb χSE] := env.edges[1,1,1][χNW DNt DNb; χNE] * ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) * env.edges[3,1,1][χSE DSt DSb; χSW]
    D, V = eigen(transfermatrix)
    
    @autoopt @tensor left[Dsvd] := env.corners[1,1,1][χ3; χ4] * env.edges[1,1,1][χ4 DNt DNb; χN] * 
    env.edges[4,1,1][χ2 DWt DWb; χ3] * ket[dt; DNt DEt DSt DWt] * conj(bra[db; DNb DEb DSb DWb]) * O₁[db; dt] * 
    env.corners[4,1,1][χ1; χ2] * env.edges[3,1,1][χS DSt DSb; χ1] * V[χN DEt DEb χS; Dsvd]

    @autoopt @tensor left_norm[Dsvd] := env.corners[1,1,1][χ3; χ4] * env.edges[1,1,1][χ4 DNt DNb; χN] * 
    env.edges[4,1,1][χ2 DWt DWb; χ3] * ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) *
    env.corners[4,1,1][χ1; χ2] * env.edges[3,1,1][χS DSt DSb; χ1] * V[χN DEt DEb χS; Dsvd]

    @autoopt @tensor right[Dsvd] := inv(V)[Dsvd; χ1 DWt DWb χ6] * 
    env.edges[1,1,1][χ1 DNt DNb; χ2] * env.corners[2,1,1][χ2; χ3] *
    ket[dt; DNt DEt DSt DWt] * conj(bra[db; DNb DEb DSb DWb]) * O₂[db; dt] * env.edges[2,1,1][χ3 DEt DEb; χ4] * 
    env.edges[3,1,1][χ5 DSt DSb; χ6] * env.corners[3,1,1][χ4; χ5]

    @autoopt @tensor right_norm[Dsvd] := inv(V)[Dsvd; χ1 DWt DWb χ6] * 
    env.edges[1,1,1][χ1 DNt DNb; χ2] * env.corners[2,1,1][χ2; χ3] *
    ket[d; DNt DEt DSt DWt] * conj(bra[d; DNb DEb DSb DWb]) * env.edges[2,1,1][χ3 DEt DEb; χ4] * 
    env.edges[3,1,1][χ5 DSt DSb; χ6] * env.corners[3,1,1][χ4; χ5]

    @tensor t[-1; -2] := left[-1] * right[-2]
    @tensor t_norm[-1; -2] := left_norm[-1] * right_norm[-2]

    function corr(r)
        c = @tensor t[1; 2] * (D^(r-1))[1; 2]
        nrm = @tensor t_norm[1; 2] * (D^(r-1))[1; 2]
        return c / nrm
    end
    return corr.(1:R)
end