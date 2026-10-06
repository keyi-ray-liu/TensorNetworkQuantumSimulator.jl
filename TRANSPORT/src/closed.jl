function _lead_fill_order(model::BuiltModel, side::Symbol)
    if model.config.lead_basis == :mixed
        modes = collect(filter(mode -> mode.side == side, model.lead_modes))
        sort!(modes; by = mode -> (mode.energy, _site_string(mode.site)))
        return getfield.(modes, :site)
    end
    count = side == :left ? model.config.n_left : model.config.n_right
    constructor = side == :left ? L : R
    return [constructor(index) for index in count:-1:1]
end

function _fill_region!(occupied, labels, particles::Tuple{Int, Int})
    length(labels) >= maximum(particles) || throw(
        ArgumentError("regional particle count exceeds the number of sites"),
    )
    for (flavor_index, flavor) in enumerate(FLAVORS)
        for site in labels[1:particles[flavor_index]]
            push!(occupied, mode_vertex(site, flavor))
        end
    end
    return occupied
end

function _product_occupations(model::BuiltModel, initial::ClosedInitialConfig)
    occupied = Set{Any}()
    _fill_region!(
        occupied,
        _lead_fill_order(model, :left),
        initial.left_particles,
    )
    _fill_region!(
        occupied,
        [S(index) for index in 1:model.config.n_system],
        initial.system_particles,
    )
    _fill_region!(
        occupied,
        _lead_fill_order(model, :right),
        initial.right_particles,
    )
    return occupied
end

function _system_onsite(config::ModelConfig, index::Int)
    values = config.system_onsite
    return values isa Tuple ? values[index] : values
end

function _site_energy(model::BuiltModel, site)
    region, index = site
    if region == :S
        return _system_onsite(model.config, index)
    end
    mode = only(filter(candidate -> candidate.site == site, model.lead_modes))
    return mode.energy
end

function _ground_state_seed(model::BuiltModel, initial::ClosedInitialConfig)
    ordered_sites = sort(
        collect(model.sites);
        by = site -> (_site_energy(model, site), _site_string(site)),
    )
    occupied = Set{Any}()
    for (flavor_index, flavor) in enumerate(FLAVORS)
        count = initial.particles[flavor_index]
        for site in ordered_sites[1:count]
            push!(occupied, mode_vertex(site, flavor))
        end
    end
    return occupied
end

function _state_from_occupations(model::BuiltModel, occupied)
    return tensornetworkstate(
        ComplexF64,
        vertex -> vertex in occupied ? "↓" : "↑",
        model.graph,
        "S=1/2",
    )
end

"""
    prepare_closed_state(config, model=build_model(config.model))

Prepare the requested product state or an approximate particle-number-seeded
ground state. Ground-state preparation uses BP-guided imaginary-time simple
update, since the parent package does not provide DMRG. Gates conserve each
flavor before truncation; see the README for the unblocked-SVD qualification.
"""
function prepare_closed_state(
        config::ClosedRunConfig,
        model::BuiltModel = build_model(config.model);
        verbose::Bool = false,
    )
    initial = config.initial
    occupied = initial.kind == :product ?
        _product_occupations(model, initial) :
        _ground_state_seed(model, initial)
    state = _state_from_occupations(model, occupied)
    cache = update(
        BeliefPropagationCache(state);
        _bp_update_kwargs(config.numerics; verbose)...,
    )

    if initial.kind == :product
        diagnostics = Dict{String, Any}(
            "kind" => "product",
            "energy" => _energy(cache, model),
            "imaginary_time_steps" => 0,
            "converged" => true,
        )
        return cache, diagnostics
    end

    previous_energy = _energy(cache, model)
    converged = false
    completed_steps = 0
    max_error = 0.0
    stable_energy_steps = 0
    last_energy_change = Inf
    for step_index in 1:config.numerics.imaginary_time_steps
        cache, step_error, _ = evolve_interval!(
            cache,
            model,
            config.numerics.imaginary_time_step;
            time_step = config.numerics.imaginary_time_step,
            numerics = config.numerics,
            imaginary = true,
            verbose = false,
        )
        completed_steps = step_index
        max_error = max(max_error, step_error)
        energy = _energy(cache, model)
        last_energy_change = abs(energy - previous_energy)
        if last_energy_change <= config.numerics.ground_state_tolerance
            stable_energy_steps += 1
        else
            stable_energy_steps = 0
        end
        if stable_energy_steps >= 3
            converged = true
            previous_energy = energy
            break
        end
        previous_energy = energy
    end
    verbose && println(
        "imaginary-time preparation: steps=$completed_steps, " *
        "energy=$previous_energy, converged=$converged",
    )
    diagnostics = Dict{String, Any}(
        "kind" => "ground_state",
        "energy" => previous_energy,
        "imaginary_time_steps" => completed_steps,
        "converged" => converged,
        "convergence_criterion" =>
            "three consecutive |delta energy| <= ground_state_tolerance",
        "last_energy_change" => last_energy_change,
        "max_truncation_error" => max_error,
    )
    return cache, diagnostics
end

function _stack_snapshots(values::Vector)
    first_value = first(values)
    first_value isa Number && return typeof(first_value).(values)
    first_array = Array(first_value)
    output = Array{eltype(first_array)}(
        undef,
        length(values),
        size(first_array)...,
    )
    trailing = ntuple(_ -> Colon(), ndims(first_array))
    for (index, value) in enumerate(values)
        output[index, trailing...] = value
    end
    return output
end

function _history_data(times, history)
    data = Dict{String, Any}("time" => Float64.(times))
    for (name, values) in history
        data[name] = _stack_snapshots(values)
    end
    return data
end

"""
    run_closed(config; write_output=true, verbose=true)

Run a fresh closed-system transport simulation. The state is evolved between
the exact requested snapshot times with a second-order Trotter circuit and a
persistent BP cache. Results are returned in memory and, by default, written to
`observables.h5` plus a Julia `final_state.jls` checkpoint.
"""
function run_closed(
        config::ClosedRunConfig;
        write_output::Bool = true,
        verbose::Bool = true,
    )
    model = build_model(config.model)
    cache, preparation = prepare_closed_state(config, model; verbose)
    times = snapshot_times(config.evolution)
    history = Dict{String, Vector{Any}}()

    function record!(time, maximum_error)
        values = measure_transport(cache, model)
        values["max_truncation_error"] = maximum_error
        for (name, value) in values
            push!(get!(history, name, Any[]), value)
        end
        if verbose
            println(
                "closed snapshot t=$(time), norm=$(values["norm"]), " *
                "χ=$(values["max_bond_dimension"])",
            )
        end
        return nothing
    end

    record!(times[1], 0.0)
    for snapshot_index in 2:length(times)
        duration = times[snapshot_index] - times[snapshot_index - 1]
        cache, maximum_error, _ = evolve_interval!(
            cache,
            model,
            duration;
            time_step = config.evolution.time_step,
            numerics = config.numerics,
            verbose = false,
        )
        record!(times[snapshot_index], maximum_error)
    end

    result = TransportResult(
        config,
        model,
        cache,
        _history_data(times, history),
        preparation,
    )
    if write_output
        save_results(result)
        save_checkpoint(result)
    end
    return result
end

run_closed(path::AbstractString; kwargs...) =
    run_closed(load_closed_config(path); kwargs...)
