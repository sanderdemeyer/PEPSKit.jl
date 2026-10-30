"""
$(TYPEDEF)

Infinite triangular lattice with a unit cell of size `(Nrows, Ncols)`.

The lattice is embedded in a matrix of sites `[r, c]`, where each site connects to the six
neighbors at offsets `(-1, 0)`, `(-1, 1)`, `(0, 1)`, `(1, 0)`, `(1, -1)` and `(0, -1)`,
corresponding to the `NW`, `NE`, `E`, `SE`, `SW` and `W` virtual legs of an
[`InfinitePEPSTriangular`](@ref). Moving along a row corresponds to the lattice vector at 0°,
and moving down a column to the lattice vector at -60°.

## Fields

$(TYPEDFIELDS)

## Constructor

    InfiniteTriangular([Nrows=1, Ncols=1])

By default, an infinite triangular lattice with a (1, 1)-unitcell is constructed.
"""
struct InfiniteTriangular <: AbstractLattice{2}
    Nrows::Int
    Ncols::Int
    function InfiniteTriangular(Nrows::Integer = 1, Ncols::Integer = 1)
        Nrows > 0 && Ncols > 0 || error("unit cell size needs to be positive")
        return new(Nrows, Ncols)
    end
end

Base.size(lattice::InfiniteTriangular) = (lattice.Nrows, lattice.Ncols)

function vertices(lattice::InfiniteTriangular)
    return CartesianIndices((1:(lattice.Nrows), 1:(lattice.Ncols)))
end

"""
$(TYPEDEF)

Infinite triangular lattice tiled by a three-site unit cell, where the three sublattices
`A`, `B` and `C` form an elementary triangle and every triangle of the lattice contains one
site of each sublattice (the √3 × √3 structure).

This tiling is not rectangular in the `[r, c]` coordinates of [`InfiniteTriangular`](@ref);
it is embedded in the `(3, 3)` cell `[A B C; C A B; B C A]`, i.e. site `[r, c]` belongs to
sublattice `mod(c - r, 3) + 1` (`1 => A`, `2 => B`, `3 => C`), see [`sublattice`](@ref). The representative sites of
`A`, `B` and `C` are `(1, 1)`, `(1, 2)` and `(2, 1)`.

Only bonds starting from the three representative sites are generated, such that the
resulting operators measure quantities per three-site unit cell. These are only meaningful
for states that have the same three-sublattice structure, e.g. an `InfinitePEPSTriangular`
with a `(3, 3)` unit cell whose tensors are repeated as `[A B C; C A B; B C A]`. Such a
state is optimized by passing `symmetrization = TriangularThreeSublattice()` to
[`fixedpoint`](@ref), see [`TriangularThreeSublattice`](@ref).

## Constructor

    InfiniteTriangularThreeSublattice()
"""
struct InfiniteTriangularThreeSublattice <: AbstractLattice{2} end

Base.size(::InfiniteTriangularThreeSublattice) = (3, 3)

vertices(::InfiniteTriangularThreeSublattice) = CartesianIndex.([(1, 1), (1, 2), (2, 1)])

"""
    sublattice(lattice::InfiniteTriangularThreeSublattice, idx::CartesianIndex{2}) -> Int

Return the sublattice (`1 => A`, `2 => B`, `3 => C`) of the site `idx`.
"""
sublattice(::InfiniteTriangularThreeSublattice, idx::CartesianIndex{2}) = mod(idx[2] - idx[1], 3) + 1

# bonds in the 0°, 60° and 120° directions, matching the triangular `energy` contractions
function nearest_neighbours(lattice::Union{InfiniteTriangular,InfiniteTriangularThreeSublattice})
    neighbors = Vector{CartesianIndex{2}}[]
    for idx in vertices(lattice)
        push!(neighbors, [idx, idx + CartesianIndex(0, 1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, -1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, 0)])
    end
    return neighbors
end

# bonds of length √3 in the 330°, 270° and 210° directions, connecting equal sublattices
function next_nearest_neighbours(lattice::Union{InfiniteTriangular,InfiniteTriangularThreeSublattice})
    neighbors = Vector{CartesianIndex{2}}[]
    for idx in vertices(lattice)
        push!(neighbors, [idx, idx + CartesianIndex(1, 1)])
        push!(neighbors, [idx, idx + CartesianIndex(2, -1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, -2)])
    end
    return neighbors
end

"""
$(TYPEDEF)

Symmetrization that ties the tensors of a `(3, 3)` [`InfinitePEPSTriangular`](@ref) to the
three-sublattice structure `[A B C; C A B; B C A]` of [`InfiniteTriangularThreeSublattice`](@ref),
by averaging all tensors on the same sublattice.

Applied to a gradient, this is the orthogonal projection onto the tied subspace, such that
`fixedpoint(H, peps, env; symmetrization = TriangularThreeSublattice())` optimizes the three
independent tensors `A`, `B` and `C`.
"""
struct TriangularThreeSublattice <: SymmetrizationStyle end

symmetrize!(peps::InfinitePEPSTriangular, ::Nothing) = peps

function symmetrize!(peps::InfinitePEPSTriangular, ::TriangularThreeSublattice)
    size(peps) == (3, 3) ||
        throw(ArgumentError("three-sublattice symmetrization requires a (3, 3) unit cell"))
    lattice = InfiniteTriangularThreeSublattice()
    for s in 1:3
        inds = filter(idx -> sublattice(lattice, idx) == s, CartesianIndices(unitcell(peps)))
        avg = sum(idx -> peps[idx], inds) / length(inds)
        for idx in inds
            peps[idx] = copy(avg)
        end
    end
    return peps
end
