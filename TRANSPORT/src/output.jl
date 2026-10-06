struct TransportResult{C, D <: AbstractDict, P <: AbstractDict}
    config::ClosedRunConfig
    model::BuiltModel
    cache::C
    data::D
    preparation::P
end

function _site_string(site)
    region, index = site
    return string(uppercase(String(region)), index)
end

function _vertex_string(vertex)
    site, flavor = vertex
    region, index = site
    return string(uppercase(String(region)), index, "_", flavor)
end

function save_results(
        result::TransportResult,
        directory::AbstractString = result.config.output_directory,
    )
    mkpath(directory)
    destination = joinpath(directory, "observables.h5")
    temporary = joinpath(directory, ".observables.h5.tmp")

    h5open(temporary, "w") do file
        metadata = attributes(file)
        metadata["system_kind"] = "closed"
        metadata["transport_model"] = String(result.config.model.particle_model)
        metadata["config_json"] = config_json(result.config)
        metadata["tnqs_version"] = string(
            Base.pkgversion(TensorNetworkQuantumSimulator),
        )
        operator_name = result.config.model.particle_model == :fermion_1d ? "c" : "b"
        metadata["current_convention"] =
            "positive left->system->right; J(i->j)=-2 Im[h_ij <$(operator_name)dag_i $(operator_name)_j>]"
        metadata["correlation_convention"] =
            "C[i,j]=<$(operator_name)dag_i $(operator_name)_j>; NN[i,j]=<n_i n_j>; system is row-major"
        metadata["norm_convention"] =
            "BP/Bethe norm unless observables were explicitly measured with alg=exact"
        metadata["checkpoint_trust"] =
            "Julia Serialization; load only files from a trusted source"
        if result.config.model.particle_model == :fermion_1d
            metadata["fermion_order"] =
                "all up modes then all down modes; each spatial chain is Ln..L1,S1..SN,R1..Rn"
        end

        write(file, "physical_sites", collect(_site_string.(result.model.sites)))
        write(file, "mode_vertices", _vertex_string.(result.model.vertices))
        for (name, value) in sort!(collect(result.data); by = first)
            write(file, name, value)
        end
        preparation_group = create_group(file, "preparation")
        preparation_metadata = attributes(preparation_group)
        for (name, value) in result.preparation
            if value isa Union{AbstractString, Number, Bool}
                preparation_metadata[name] = value
            else
                preparation_metadata[name] = JSON3.write(value)
            end
        end
    end
    mv(temporary, destination; force = true)
    return destination
end

function save_checkpoint(
        result::TransportResult,
        directory::AbstractString = result.config.output_directory,
    )
    mkpath(directory)
    destination = joinpath(directory, "final_state.jls")
    temporary = joinpath(directory, ".final_state.jls.tmp")
    payload = (
        format_version = 1,
        config = result.config,
        state = network(result.cache),
        data = result.data,
        preparation = result.preparation,
    )
    open(temporary, "w") do stream
        serialize(stream, payload)
    end
    mv(temporary, destination; force = true)
    return destination
end

function load_checkpoint(path::AbstractString)
    checkpoint_path = isdir(path) ? joinpath(path, "final_state.jls") : path
    payload = open(deserialize, checkpoint_path)
    payload.format_version == 1 || error(
        "unsupported BPTransport checkpoint version $(payload.format_version)",
    )
    model = build_model(payload.config.model)
    cache = BeliefPropagationCache(payload.state)
    # A freshly deserialized state has no valid BP messages. Always rebuild
    # them before exposing a cache so immediate measurements are sound.
    cache = update(
        cache;
        _bp_update_kwargs(payload.config.numerics)...,
    )
    return TransportResult(
        payload.config,
        model,
        cache,
        payload.data,
        payload.preparation,
    )
end
