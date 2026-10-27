function _contract_edges_0((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 2, 2}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    O_r_cn = network[r, _next(c, end)]
    return @tensor opt = true ket(O_r_c)[dtL; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(O_r_cn)[dtR; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(O_r_c)[dbL; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(O_r_cn)[dbR; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1, _prev(r, end), c][χNW DLt120 DLb120; χaN] * env.C[2, _prev(r, end), _next(c + 1, end)][χbN DRt60 DRb60; χNE] * env.C[3, r, _next(c + 1, end)][χNE DRt0 DRb0; χSE] *
        env.C[4, _next(r, end), _next(c, end)][χSE DRt300 DRb300; χaS] * env.C[5, _next(r, end), _prev(c, end)][χbS DLt240 DLb240; χSW] * env.C[6, r, _prev(c, end)][χSW DLt180 DLb180; χNW] *
        env.Eb[1, _prev(r, end), _next(c, end)][χaN DLt60 DLb60; χCN] * env.Ea[1, _prev(r, end), _next(c, end)][χCN DRt120 DRb120; χbN] *
        env.Eb[4, _next(r, end), c][χaS DRt240 DRb240; χCS] * env.Ea[4, _next(r, end), c][χCS DLt300 DLb300; χbS] *
        op[dbL dbR; dtL dtR]
end

function _contract_corners((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 1, 1}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    return @tensor opt = true env.C[1, _prev(r, end), c][χNW Dt120 Db120; χN] * env.C[2, _prev(r, end), _next(c, end)][χN Dt60 Db60; χNE] * env.C[3, r, _next(c, end)][χNE Dt0 Db0; χSE] *
        env.C[4, _next(r, end), c][χSE Dt300 Db300; χS] * env.C[5, _next(r, end), _prev(c, end)][χS Dt240 Db240; χSW] * env.C[6, r, _prev(c, end)][χSW Dt180 Db180; χNW] *
        ket(O_r_c)[dt; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] * conj(bra(O_r_c)[db; Db120 Db60 Db0 Db300 Db240 Db180]) * op[db; dt]
end
