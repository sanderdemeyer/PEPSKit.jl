"""
    InfiniteSquareNetwork{O}

Contractible square network. Wraps a matrix of 'rank-4-tensor-like' objects.
"""
struct InfiniteSquareNetwork{O}
    A::Matrix{O}
    InfiniteSquareNetwork{O}(A::Matrix{O}) where {O} = new{O}(A)
    function InfiniteSquareNetwork(A::Array{O,2}) where {O}
        for (d, w) in Tuple.(CartesianIndices(A))
            north_virtualspace(A[d, w]) ==
            _elementwise_dual(south_virtualspace(A[_prev(d, end), w])) || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            east_virtualspace(A[d, w]) ==
            _elementwise_dual(west_virtualspace(A[d, _next(w, end)])) ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        end
        return new{O}(A)
    end
end

## Shape and size
unitcell(n::InfiniteSquareNetwork) = n.A
Base.size(n::InfiniteSquareNetwork, args...) = size(unitcell(n), args...)
Base.length(n::InfiniteSquareNetwork) = length(unitcell(n))
Base.eltype(n::InfiniteSquareNetwork) = eltype(typeof(n))
Base.eltype(::Type{<:InfiniteSquareNetwork{O}}) where {O} = O

## Copy
Base.copy(n::InfiniteSquareNetwork) = InfiniteSquareNetwork(copy(unitcell(n)))
function Base.similar(n::InfiniteSquareNetwork, args...)
    return InfiniteSquareNetwork(similar(unitcell(n), args...))
end
function Base.repeat(n::InfiniteSquareNetwork, counts...)
    return InfiniteSquareNetwork(repeat(unitcell(n), counts...))
end

## Indexing
function Base.getindex(n::InfiniteSquareNetwork, args...)
    return Base.getindex(unitcell(n), args...)
end
function Base.setindex!(n::InfiniteSquareNetwork, args...)
    return (Base.setindex!(unitcell(n), args...); n)
end
Base.axes(n::InfiniteSquareNetwork, args...) = axes(unitcell(n), args...)
function eachcoordinate(n::InfiniteSquareNetwork)
    return collect(Iterators.product(axes(n)...))
end
function eachcoordinate(n::InfiniteSquareNetwork, dirs)
    return collect(Iterators.product(dirs, axes(n, 1), axes(n, 2)))
end

## Spaces
virtualspace(n::InfiniteSquareNetwork, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)

## Vector interface
VectorInterface.scalartype(::Type{<:InfiniteSquareNetwork{O}}) where {O} = scalartype(O)
function VectorInterface.zerovector(A::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(zerovector(unitcell(A)))
end

## Math (for Zygote accumulation)
function Base.:+(A₁::NWType, A₂::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A₁) .+ unitcell(A₂))
end
function Base.:-(A₁::NWType, A₂::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A₁) .- unitcell(A₂))
end
function Base.:*(α::Number, A::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(α .* unitcell(A))
end
function Base.:/(A::NWType, α::Number) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A) ./ α)
end
function LinearAlgebra.dot(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return dot(unitcell(A₁), unitcell(A₂))
end
LinearAlgebra.norm(A::InfiniteSquareNetwork) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations
function Base.rotl90(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rotl90(rotl90.(unitcell(n))))
end
function Base.rotr90(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rotr90(rotr90.(unitcell(n))))
end
function Base.rot180(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rot180(rot180.(unitcell(n))))
end

## Chainrules
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), network::InfiniteSquareNetwork, args...
)
    O = network[args...]

    function getindex_pullback(ΔO_)
        ΔO = unthunk(ΔO_)
        if ΔO isa Tangent # TODO: figure out why this happens in the first place...
            ΔO = ChainRulesCore.construct(typeof(O), ChainRulesCore.backing(ΔO))
        end
        Δnetwork = zerovector(network)
        Δnetwork[args...] = ΔO
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return O, getindex_pullback
end

# TODO: not actually used?
function ChainRulesCore.rrule(
    ::Type{NWType}, A::Matrix
) where {NWType<:InfiniteSquareNetwork}
    network = NWType(A)
    function InfiniteSquareNetwork_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), unnitcell(Δnetwork)
    end
    return network, InfiniteSquareNetwork_pullback
end

function ChainRulesCore.rrule(::typeof(rotl90), network::InfiniteSquareNetwork)
    network´ = rotl90(network)
    function rotl90_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), rotr90(Δnetwork)
    end
    return network´, rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), network::InfiniteSquareNetwork)
    network´ = rotr90(network)
    function rotr90_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), rotl90(Δnetwork)
    end
    return network´, rotr90_pullback
end
