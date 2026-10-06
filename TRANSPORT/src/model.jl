"""Canonical labels for the left lead, central system, and right lead."""
L(k::Integer) = (:L, Int(k))
S(k::Integer) = (:S, Int(k))
R(k::Integer) = (:R, Int(k))

const FLAVORS = (:up, :down)
const PhysicalSite = Tuple{Symbol, Int}
const ModeVertex = Tuple{PhysicalSite, Symbol}

"""
    mode_vertex(site, flavor)

Return the tensor-network vertex for `flavor` on a physical transport site.
The transport graph has one qubit layer per flavor.
"""
function mode_vertex(site::PhysicalSite, flavor::Symbol)
    flavor in FLAVORS || throw(ArgumentError("unknown flavor $flavor; expected one of $FLAVORS"))
    return (site, flavor)
end

"""A lead orbital together with its onsite energy and device-contact amplitude."""
struct LeadMode
    site::PhysicalSite
    side::Symbol
    index::Int
    energy::Float64
    contact::Float64
end

"""A physical contact oriented in the positive (left-to-right) transport direction."""
struct Contact
    side::Symbol
    origin::PhysicalSite
    destination::PhysicalSite
    amplitude::ComplexF64
end

"""An onsite Hamiltonian contribution `amplitude * n_vertex`."""
struct OnsiteTerm
    vertex::ModeVertex
    amplitude::ComplexF64
    kind::Symbol
end

"""
A two-mode Hamiltonian contribution.

For `kind == :hopping`, the convention is
`amplitude * b†_origin * b_destination + conj(amplitude) * b†_destination * b_origin`.
For `:hubbard` and `:density`, it is
`amplitude * n_origin * n_destination`.
"""
struct BondTerm
    origin::ModeVertex
    destination::ModeVertex
    amplitude::ComplexF64
    kind::Symbol
end

"""Graph geometry and local Hamiltonian terms for a BP transport calculation."""
struct BuiltModel{C}
    config::C
    graph::NamedGraph{ModeVertex}
    sites::Tuple
    vertices::Vector{ModeVertex}
    site_to_index::Dictionary{PhysicalSite, Int}
    onsite_terms::Vector{OnsiteTerm}
    bond_terms::Vector{BondTerm}
    contacts::Tuple
    lead_modes::Vector{LeadMode}
    dummy_edges::Vector{NamedEdge{ModeVertex}}
end

_config_symbol(value::Symbol) = value
_config_symbol(value::AbstractString) = Symbol(lowercase(value))

function _side_symbol(side)
    value = _config_symbol(side)
    value in (:left, :right) ||
        throw(ArgumentError("side must be :left or :right, received $side"))
    return value
end

"""
    lead_mode(config, side, k) -> LeadMode

Construct lead mode `k` using the position, sine-grid, or logarithmic-grid
conventions of the reference transport implementation.
"""
function lead_mode(config::ModelConfig, side, k::Integer)
    side = _side_symbol(side)
    n = Int(side == :left ? config.n_left : config.n_right)
    1 <= k <= n || throw(ArgumentError("lead mode index $k is outside 1:$n"))

    potential = Float64(
        side == :left ? config.lead_potential_left : config.lead_potential_right,
    )
    coupling = Float64(side == :left ? config.coupling_left : config.coupling_right)
    site = side == :left ? L(k) : R(k)
    basis = _config_symbol(config.lead_basis)

    if basis == :position
        contact = k == 1 ? coupling : 0.0
        return LeadMode(site, side, Int(k), potential, contact)
    elseif basis != :mixed
        throw(ArgumentError("lead_basis must be :position or :mixed, received $(config.lead_basis)"))
    end

    grid = _config_symbol(config.lead_grid)
    hopping = Float64(config.lead_hopping)
    if grid == :sine
        angle = π * k / (n + 1)
        energy = potential + 2 * hopping * cos(angle)
        contact = coupling * sin(angle) * sqrt(2 / (n + 1))
    elseif grid == :log
        iseven(n) || throw(ArgumentError("logarithmic lead grids require an even number of modes"))
        isnothing(config.log_lambda) &&
            throw(ArgumentError("logarithmic lead grids require log_lambda"))
        λ = Float64(config.log_lambda)
        λ > 1 || throw(ArgumentError("log_lambda must be greater than one"))
        effective_k = k <= n ÷ 2 ? k : n + 1 - k
        sign_k = k <= n ÷ 2 ? 1.0 : -1.0
        energy = potential + 2 * hopping * sign_k * λ^(-effective_k + 0.5)
        contact = coupling * sqrt(2 / π * (1 - inv(λ)) * λ^(-effective_k + 1))
    else
        throw(ArgumentError("lead_grid must be :sine or :log, received $(config.lead_grid)"))
    end

    return LeadMode(site, side, Int(k), Float64(energy), Float64(contact))
end

"""Return all lead modes in physical output order: left lead, then right lead."""
function all_lead_modes(config::ModelConfig)
    left = [lead_mode(config, :left, k) for k in 1:Int(config.n_left)]
    right = [lead_mode(config, :right, k) for k in 1:Int(config.n_right)]
    return vcat(left, right)
end

"""Return physical sites in output order `L(1:n_left), S(1:n_system), R(1:n_right)`."""
function canonical_sites(config::ModelConfig)
    rows, columns = Int.(Tuple(config.system_shape))
    n_system = rows * columns
    return Tuple(
        vcat(
            [L(k) for k in 1:Int(config.n_left)],
            [S(k) for k in 1:n_system],
            [R(k) for k in 1:Int(config.n_right)],
        ),
    )
end

function _system_onsite_values(config::ModelConfig, n_system::Int)
    values = config.system_onsite
    if values isa Number
        return fill(Float64(values), n_system)
    end
    output = Float64.(collect(values))
    length(output) == n_system || throw(
        ArgumentError("system_onsite must be scalar or have $n_system entries"),
    )
    return output
end

function _contacts(config::ModelConfig, modes::Vector{LeadMode}, n_system::Int)
    basis = _config_symbol(config.lead_basis)
    if basis == :position
        return (
            Contact(:left, L(1), S(1), ComplexF64(config.coupling_left)),
            Contact(:right, S(n_system), R(1), ComplexF64(config.coupling_right)),
        )
    end

    left = Contact[
        Contact(:left, mode.site, S(1), ComplexF64(mode.contact))
        for mode in modes if mode.side == :left
    ]
    right = Contact[
        Contact(:right, S(n_system), mode.site, ComplexF64(mode.contact))
        for mode in modes if mode.side == :right
    ]
    return Tuple(vcat(left, right))
end

"""
    build_model(config::ModelConfig) -> BuiltModel

Build the bounded-local-dimension, two-flavor hard-core-particle model used by
the BP transport drivers. Every Hamiltonian bond is also an edge of the tensor
network graph. Zero-strength hopping geometry is retained as dimension-one
structural edges. If the Hamiltonian has no cross-flavor interaction, one
dimension-one dummy edge joins the two flavor layers without adding a term.
"""
function build_model(config::ModelConfig)
    n_left, n_right = Int(config.n_left), Int(config.n_right)
    n_left >= 1 || throw(ArgumentError("n_left must be positive"))
    n_right >= 1 || throw(ArgumentError("n_right must be positive"))
    shape = Tuple(config.system_shape)
    length(shape) == 2 || throw(ArgumentError("system_shape must have two entries"))
    rows, columns = Int.(shape)
    rows >= 1 && columns >= 1 ||
        throw(ArgumentError("system_shape entries must be positive"))
    n_system = rows * columns

    sites = canonical_sites(config)
    mode_vertices = ModeVertex[
        mode_vertex(site, flavor) for site in sites for flavor in FLAVORS
    ]
    vertex_position = Dict(vertex => i for (i, vertex) in enumerate(mode_vertices))
    site_to_index = Dictionary(collect(sites), collect(eachindex(sites)))

    onsite_terms = OnsiteTerm[]
    bond_terms = BondTerm[]
    graph_edges = NamedEdge{ModeVertex}[]
    edge_keys = Set{Tuple{Int, Int}}()

    edge_key(a::ModeVertex, b::ModeVertex) = begin
        ia, ib = vertex_position[a], vertex_position[b]
        ia < ib ? (ia, ib) : (ib, ia)
    end

    function add_graph_edge!(a::ModeVertex, b::ModeVertex)
        a == b && throw(ArgumentError("self edges are not valid transport graph bonds"))
        key = edge_key(a, b)
        if !(key in edge_keys)
            push!(edge_keys, key)
            push!(graph_edges, NamedEdge{ModeVertex}(a => b))
        end
        return nothing
    end

    function add_onsite!(vertex::ModeVertex, amplitude, kind::Symbol = :potential)
        iszero(amplitude) ||
            push!(onsite_terms, OnsiteTerm(vertex, ComplexF64(amplitude), kind))
        return nothing
    end

    function add_bond!(
            origin::ModeVertex,
            destination::ModeVertex,
            amplitude,
            kind::Symbol;
            structural::Bool = false,
        )
        (structural || !iszero(amplitude)) && add_graph_edge!(origin, destination)
        iszero(amplitude) || push!(
            bond_terms,
            BondTerm(origin, destination, ComplexF64(amplitude), kind),
        )
        return nothing
    end

    modes = all_lead_modes(config)
    contacts = _contacts(config, modes, n_system)
    system_onsite = _system_onsite_values(config, n_system)

    # Number terms on all lead orbitals and central-system modes.
    for flavor in FLAVORS
        for mode in modes
            add_onsite!(mode_vertex(mode.site, flavor), mode.energy)
        end
        for (k, amplitude) in enumerate(system_onsite)
            add_onsite!(mode_vertex(S(k), flavor), amplitude)
        end
    end

    # Each flavor has an identical hopping layer.
    for flavor in FLAVORS
        if _config_symbol(config.lead_basis) == :position
            for k in 1:(n_left - 1)
                add_bond!(
                    mode_vertex(L(k), flavor),
                    mode_vertex(L(k + 1), flavor),
                    config.lead_hopping,
                    :hopping;
                    structural = true,
                )
            end
            for k in 1:(n_right - 1)
                add_bond!(
                    mode_vertex(R(k), flavor),
                    mode_vertex(R(k + 1), flavor),
                    config.lead_hopping,
                    :hopping;
                    structural = true,
                )
            end
        end

        for contact in contacts
            add_bond!(
                mode_vertex(contact.origin, flavor),
                mode_vertex(contact.destination, flavor),
                contact.amplitude,
                :hopping;
                structural = true,
            )
        end

        for row in 1:rows, column in 1:columns
            k = (row - 1) * columns + column
            if row < rows
                add_bond!(
                    mode_vertex(S(k), flavor),
                    mode_vertex(S(k + columns), flavor),
                    config.system_hopping,
                    :hopping;
                    structural = true,
                )
            end
            if column < columns
                add_bond!(
                    mode_vertex(S(k), flavor),
                    mode_vertex(S(k + 1), flavor),
                    config.system_hopping,
                    :hopping;
                    structural = true,
                )
            end
        end
    end

    # Onsite Hubbard interaction becomes a density rung between flavor qubits.
    if !iszero(config.hubbard_u)
        for k in 1:n_system
            add_bond!(
                mode_vertex(S(k), :up),
                mode_vertex(S(k), :down),
                config.hubbard_u,
                :hubbard,
            )
        end
    end

    # Standard intersite total-density interaction, expanded over all flavors.
    if !iszero(config.density_interaction)
        softening = Float64(config.interaction_softening)
        softening > 0 || throw(ArgumentError("interaction_softening must be positive"))
        for i in 1:n_system
            ri, ci = divrem(i - 1, columns)
            for j in (i + 1):n_system
                rj, cj = divrem(j - 1, columns)
                distance = hypot(ri - rj, ci - cj)
                amplitude = config.density_interaction / (distance + softening)
                for flavor_i in FLAVORS, flavor_j in FLAVORS
                    add_bond!(
                        mode_vertex(S(i), flavor_i),
                        mode_vertex(S(j), flavor_j),
                        amplitude,
                        :density,
                    )
                end
            end
        end
    end

    # With no physical cross-flavor coupling, connect the two TN components by
    # one untouched dimension-one bond so TNQS can maintain a connected cache.
    dummy_edges = NamedEdge{ModeVertex}[]
    layers_connected = any(
        term -> term.origin[2] != term.destination[2],
        bond_terms,
    )
    if !layers_connected
        dummy = NamedEdge{ModeVertex}(
            mode_vertex(S(1), :up) => mode_vertex(S(1), :down),
        )
        add_graph_edge!(dummy.src, dummy.dst)
        push!(dummy_edges, dummy)
    end

    # Materialize the graph only after de-duplicating physical and structural
    # edges. This permits hopping and density terms to share one TN bond.
    graph = NamedGraph(mode_vertices)
    for edge in graph_edges
        graph = add_edge(graph, edge)
    end

    # Internal invariant: every gate-generating two-site term is graph-local.
    @assert all(
        edge_key(term.origin, term.destination) in edge_keys for term in bond_terms
    )

    return BuiltModel(
        config,
        graph,
        sites,
        mode_vertices,
        site_to_index,
        onsite_terms,
        bond_terms,
        contacts,
        modes,
        dummy_edges,
    )
end
