function _expectation(source, observable; alg::Union{Nothing, String} = nothing)
    if source isa BeliefPropagationCache
        if isnothing(alg) || alg == "bp"
            return expect(source, observable)
        end
        return expect(network(source), observable; alg)
    end
    return expect(source, observable; alg = something(alg, "bp"))
end

function _occupation(source, vertex; alg = nothing)
    z = _expectation(source, ("Z", [vertex]); alg)
    return (1 - real(z)) / 2
end

function _hardcore_correlator(source, origin, destination; alg = nothing)
    origin == destination && return complex(_occupation(source, origin; alg))
    return _expectation(
        source,
        (["S-", "S+"], [origin, destination]);
        alg,
    )
end

# Jordan--Wigner order for the explicitly supported one-dimensional fermion
# interpretation. The spatial chain runs from the remote left boundary through
# the device to the remote right boundary; the complete up chain precedes the
# complete down chain. The latter choice fixes cross-flavor Klein-factor signs.
function _fermion_mode_order(model::BuiltModel)
    config = model.config
    spatial_order = vcat(
        [L(k) for k in config.n_left:-1:1],
        [S(k) for k in 1:config.n_system],
        [R(k) for k in 1:config.n_right],
    )
    return [mode_vertex(site, flavor) for flavor in FLAVORS for site in spatial_order]
end

function _fermion_correlator(source, origin, destination, model; alg = nothing)
    origin == destination && return complex(_occupation(source, origin; alg))
    order = _fermion_mode_order(model)
    origin_index = findfirst(==(origin), order)
    destination_index = findfirst(==(destination), order)
    isnothing(origin_index) && throw(ArgumentError("unknown mode vertex $origin"))
    isnothing(destination_index) && throw(ArgumentError("unknown mode vertex $destination"))

    if origin_index > destination_index
        return conj(_fermion_correlator(source, destination, origin, model; alg))
    end

    vertices = order[origin_index:destination_index]
    operators = vcat("S-", fill("Z", length(vertices) - 2), "S+")
    return _expectation(source, (operators, vertices); alg)
end

function _correlator(source, origin, destination, model::BuiltModel; alg = nothing)
    if model.config.particle_model == :fermion_1d
        return _fermion_correlator(source, origin, destination, model; alg)
    end
    return _hardcore_correlator(source, origin, destination; alg)
end

function _density_correlation(source, first_vertex, second_vertex; alg = nothing)
    first_vertex == second_vertex && return _occupation(source, first_vertex; alg)
    first_z = real(_expectation(source, ("Z", [first_vertex]); alg))
    second_z = real(_expectation(source, ("Z", [second_vertex]); alg))
    zz = real(
        _expectation(source, ("ZZ", [first_vertex, second_vertex]); alg),
    )
    return (1 - first_z - second_z + zz) / 4
end

function _norm_squared(source; alg = nothing)
    if source isa BeliefPropagationCache
        if isnothing(alg) || alg == "bp"
            return norm_sqr(source; alg = "bp")
        end
        return norm_sqr(network(source); alg)
    end
    return norm_sqr(source; alg = something(alg, "bp"))
end

function _energy(source, model::BuiltModel; alg = nothing)
    value = 0.0 + 0.0im
    for term in model.onsite_terms
        value += term.amplitude * _occupation(source, term.vertex; alg)
    end
    for term in model.bond_terms
        if term.kind == :hopping
            correlation = _correlator(
                source,
                term.origin,
                term.destination,
                model;
                alg,
            )
            value += term.amplitude * correlation +
                conj(term.amplitude * correlation)
        elseif term.kind in (:hubbard, :density)
            value += term.amplitude *
                _density_correlation(source, term.origin, term.destination; alg)
        else
            throw(ArgumentError("unsupported bond-term kind $(repr(term.kind))"))
        end
    end
    return real(value)
end

function _system_correlations(source, model::BuiltModel; alg = nothing)
    count = model.config.n_system
    c_up = zeros(ComplexF64, count, count)
    c_down = zeros(ComplexF64, count, count)
    c_up_down = zeros(ComplexF64, count, count)
    nn_up = zeros(Float64, count, count)
    nn_down = zeros(Float64, count, count)
    nn_up_down = zeros(Float64, count, count)

    for (flavor, c_matrix, nn_matrix) in
            ((:up, c_up, nn_up), (:down, c_down, nn_down))
        for i in 1:count, j in i:count
            vi = mode_vertex(S(i), flavor)
            vj = mode_vertex(S(j), flavor)
            c_matrix[i, j] = _correlator(source, vi, vj, model; alg)
            c_matrix[j, i] = conj(c_matrix[i, j])
            nn_matrix[i, j] = _density_correlation(source, vi, vj; alg)
            nn_matrix[j, i] = nn_matrix[i, j]
        end
    end

    for i in 1:count, j in 1:count
        up = mode_vertex(S(i), :up)
        down = mode_vertex(S(j), :down)
        c_up_down[i, j] = _correlator(source, up, down, model; alg)
        nn_up_down[i, j] = _density_correlation(source, up, down; alg)
    end

    return (; c_up, c_down, c_up_down, nn_up, nn_down, nn_up_down)
end

"""
    measure_transport(state_or_cache, model; alg=nothing,
                      include_correlations=true)

Measure spin-resolved occupations, oriented contact currents, energy, norm, and
device correlations. Passing an updated `BeliefPropagationCache` uses its
messages directly. Set `alg="exact"` on tiny systems for a contraction-based
reference.
"""
function measure_transport(
        source,
        model::BuiltModel;
        alg::Union{Nothing, String} = nothing,
        include_correlations::Bool = true,
    )
    occupations = Dict{Symbol, Vector{Float64}}()
    currents = Dict(:up => zeros(2), :down => zeros(2))
    side_index = Dict(:left => 1, :right => 2)

    for flavor in FLAVORS
        occupations[flavor] = [
            _occupation(source, mode_vertex(site, flavor); alg)
            for site in model.sites
        ]
        for contact in model.contacts
            origin = mode_vertex(contact.origin, flavor)
            destination = mode_vertex(contact.destination, flavor)
            correlation = _correlator(source, origin, destination, model; alg)
            currents[flavor][side_index[contact.side]] +=
                -2 * imag(contact.amplitude * correlation)
        end
    end

    values = Dict{String, Any}(
        "occupation_up" => occupations[:up],
        "occupation_down" => occupations[:down],
        "current_up" => currents[:up],
        "current_down" => currents[:down],
        "total_up" => sum(occupations[:up]),
        "total_down" => sum(occupations[:down]),
        "norm" => sqrt(max(0.0, real(_norm_squared(source; alg)))),
        "energy" => _energy(source, model; alg),
        "max_bond_dimension" => maxvirtualdim(source),
    )

    if include_correlations
        correlations = _system_correlations(source, model; alg)
        values["system_cdag_c_up"] = correlations.c_up
        values["system_cdag_c_down"] = correlations.c_down
        values["system_cdag_c_up_down"] = correlations.c_up_down
        values["system_nn_up"] = correlations.nn_up
        values["system_nn_down"] = correlations.nn_down
        values["system_nn_up_down"] = correlations.nn_up_down
    end
    return values
end
