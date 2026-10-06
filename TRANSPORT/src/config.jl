"""Strict configuration types and JSON loading for closed BP transport runs."""

import JSON3

const _SystemOnsite = Union{Float64,Tuple{Vararg{Float64}}}


struct ModelConfig
    n_left::Int
    n_right::Int
    system_shape::Tuple{Int,Int}
    lead_basis::Symbol
    lead_grid::Symbol
    log_lambda::Union{Nothing,Float64}
    lead_hopping::Float64
    system_hopping::Float64
    coupling_left::Float64
    coupling_right::Float64
    lead_potential_left::Float64
    lead_potential_right::Float64
    system_onsite::_SystemOnsite
    hubbard_u::Float64
    density_interaction::Float64
    interaction_softening::Float64
    particle_model::Symbol
end

function ModelConfig(;
        n_left,
        n_right,
        system_shape,
        lead_basis = :position,
        lead_grid = :sine,
        log_lambda = nothing,
        lead_hopping = 1.0,
        system_hopping = 1.0,
        coupling_left = 0.2,
        coupling_right = 0.2,
        lead_potential_left = 0.0,
        lead_potential_right = 0.0,
        system_onsite = 0.0,
        hubbard_u = 0.0,
        density_interaction = 0.0,
        interaction_softening = 0.5,
        particle_model = :hardcore,
    )
    nl = _config_int(n_left, "n_left")
    nr = _config_int(n_right, "n_right")
    shape = _config_int_pair(system_shape, "system_shape"; allow_integral_reals = true)
    basis = _config_symbol(lead_basis, "lead_basis")
    grid = _config_symbol(lead_grid, "lead_grid")
    lambda = isnothing(log_lambda) ? nothing : _config_float(log_lambda, "log_lambda")
    onsite = _config_system_onsite(system_onsite)
    particles = _config_symbol(particle_model, "particle_model")

    nl > 0 || throw(ArgumentError("n_left must be positive"))
    nr > 0 || throw(ArgumentError("n_right must be positive"))
    all(>(0), shape) || throw(ArgumentError("system_shape entries must be positive"))
    basis in (:position, :mixed) ||
        throw(ArgumentError("lead_basis must be :position or :mixed"))
    grid in (:sine, :log) || throw(ArgumentError("lead_grid must be :sine or :log"))

    if basis == :position && (grid != :sine || !isnothing(lambda))
        throw(ArgumentError("lead_grid and log_lambda apply only to mixed leads"))
    end
    if grid == :sine && !isnothing(lambda)
        throw(ArgumentError("log_lambda is valid only when lead_grid is :log"))
    end
    if grid == :log
        !isnothing(lambda) && lambda > 1 ||
            throw(ArgumentError("log grids require log_lambda > 1"))
        iseven(nl) && iseven(nr) ||
            throw(ArgumentError("log grids require even lead sizes"))
    end

    nsys = prod(shape)
    if onsite isa Tuple && length(onsite) != nsys
        throw(ArgumentError("system_onsite must be scalar or have n_system entries"))
    end

    softening = _config_float(interaction_softening, "interaction_softening")
    softening > 0 || throw(ArgumentError("interaction_softening must be positive"))

    particles in (:hardcore, :fermion_1d) ||
        throw(ArgumentError("particle_model must be :hardcore or :fermion_1d"))
    if particles == :fermion_1d && (basis != :position || !(1 in shape))
        throw(ArgumentError(
            "particle_model=:fermion_1d requires position-basis leads and a one-dimensional system",
        ))
    end

    return ModelConfig(
        nl,
        nr,
        shape,
        basis,
        grid,
        lambda,
        _config_float(lead_hopping, "lead_hopping"),
        _config_float(system_hopping, "system_hopping"),
        _config_float(coupling_left, "coupling_left"),
        _config_float(coupling_right, "coupling_right"),
        _config_float(lead_potential_left, "lead_potential_left"),
        _config_float(lead_potential_right, "lead_potential_right"),
        onsite,
        _config_float(hubbard_u, "hubbard_u"),
        _config_float(density_interaction, "density_interaction"),
        softening,
        particles,
    )
end


"""Number of interacting-system sites in `config`."""
n_system(config::ModelConfig) = prod(config.system_shape)

"""Total number of logical orbitals in `config`."""
n_sites(config::ModelConfig) = config.n_left + n_system(config) + config.n_right

# Match the reference configuration's computed-property interface while keeping
# the stored configuration immutable and free of redundant derived fields.
function Base.getproperty(config::ModelConfig, name::Symbol)
    name === :n_system && return prod(getfield(config, :system_shape))
    name === :n_sites && return (
        getfield(config, :n_left) + prod(getfield(config, :system_shape)) +
        getfield(config, :n_right)
    )
    return getfield(config, name)
end

function Base.propertynames(config::ModelConfig, private::Bool = false)
    stored = fieldnames(typeof(config))
    return (stored..., :n_system, :n_sites)
end

"""Return one onsite energy per system site."""
function onsite_values(config::ModelConfig)
    if config.system_onsite isa Float64
        return ntuple(_ -> config.system_onsite, n_system(config))
    end
    return config.system_onsite
end


struct EvolutionConfig
    final_time::Float64
    snapshot_interval::Float64
    time_step::Float64
end

function EvolutionConfig(; final_time, snapshot_interval, time_step)
    final = _config_float(final_time, "final_time")
    snapshot = _config_float(snapshot_interval, "snapshot_interval")
    step = _config_float(time_step, "time_step")
    final >= 0 || throw(ArgumentError("final_time must be non-negative"))
    snapshot > 0 || throw(ArgumentError("snapshot_interval must be positive"))
    step > 0 || throw(ArgumentError("time_step must be positive"))
    return EvolutionConfig(final, snapshot, step)
end


struct NumericsConfig
    bond_dimension::Int
    svd_tolerance::Float64
    bp_maxiter::Int
    bp_tolerance::Float64
    imaginary_time_steps::Int
    imaginary_time_step::Float64
    ground_state_tolerance::Float64
end

function NumericsConfig(;
        bond_dimension = 64,
        svd_tolerance = 1.0e-9,
        bp_maxiter = 25,
        bp_tolerance = 1.0e-8,
        imaginary_time_steps = 100,
        imaginary_time_step = 0.02,
        ground_state_tolerance = 1.0e-8,
    )
    bond = _config_int(bond_dimension, "bond_dimension")
    maxiter = _config_int(bp_maxiter, "bp_maxiter")
    imaginary_steps = _config_int(imaginary_time_steps, "imaginary_time_steps")
    svd_tol = _config_float(svd_tolerance, "svd_tolerance")
    bp_tol = _config_float(bp_tolerance, "bp_tolerance")
    imaginary_step = _config_float(imaginary_time_step, "imaginary_time_step")
    ground_tol = _config_float(ground_state_tolerance, "ground_state_tolerance")

    bond > 0 || throw(ArgumentError("bond_dimension must be positive"))
    maxiter > 0 || throw(ArgumentError("bp_maxiter must be positive"))
    imaginary_steps >= 0 ||
        throw(ArgumentError("imaginary_time_steps must be non-negative"))
    svd_tol > 0 || throw(ArgumentError("svd_tolerance must be positive"))
    bp_tol > 0 || throw(ArgumentError("bp_tolerance must be positive"))
    imaginary_step > 0 || throw(ArgumentError("imaginary_time_step must be positive"))
    ground_tol > 0 || throw(ArgumentError("ground_state_tolerance must be positive"))

    return NumericsConfig(
        bond,
        svd_tol,
        maxiter,
        bp_tol,
        imaginary_steps,
        imaginary_step,
        ground_tol,
    )
end


struct ClosedInitialConfig
    kind::Symbol
    particles::Tuple{Int,Int}
    left_particles::Tuple{Int,Int}
    system_particles::Tuple{Int,Int}
    right_particles::Tuple{Int,Int}
end

function ClosedInitialConfig(;
        kind = :ground_state,
        particles = nothing,
        left_particles = nothing,
        system_particles = nothing,
        right_particles = nothing,
    )
    initial_kind = _config_symbol(kind, "kind")
    initial_kind in (:product, :ground_state) ||
        throw(ArgumentError("closed initial kind must be :product or :ground_state"))

    if initial_kind == :product
        isnothing(particles) ||
            throw(ArgumentError("product initialization does not use particles"))
    elseif any(!isnothing, (left_particles, system_particles, right_particles))
        throw(ArgumentError("ground-state initialization uses only total particles"))
    end

    total = isnothing(particles) ? (1, 1) :
        _config_int_pair(particles, "particles"; allow_integral_reals = true)
    left = isnothing(left_particles) ? (0, 0) :
        _config_int_pair(left_particles, "left_particles"; allow_integral_reals = true)
    system = isnothing(system_particles) ? (0, 0) :
        _config_int_pair(system_particles, "system_particles"; allow_integral_reals = true)
    right = isnothing(right_particles) ? (0, 0) :
        _config_int_pair(right_particles, "right_particles"; allow_integral_reals = true)
    return ClosedInitialConfig(initial_kind, total, left, system, right)
end


struct ClosedRunConfig
    model::ModelConfig
    initial::ClosedInitialConfig
    evolution::EvolutionConfig
    numerics::NumericsConfig
    output_directory::String
end

function ClosedRunConfig(;
        model,
        initial,
        evolution,
        numerics = NumericsConfig(),
        output_directory = "closed_results",
    )
    model isa ModelConfig || throw(ArgumentError("model must be a ModelConfig"))
    initial isa ClosedInitialConfig ||
        throw(ArgumentError("initial must be a ClosedInitialConfig"))
    evolution isa EvolutionConfig ||
        throw(ArgumentError("evolution must be an EvolutionConfig"))
    numerics isa NumericsConfig ||
        throw(ArgumentError("numerics must be a NumericsConfig"))
    output_directory isa AbstractString ||
        throw(ArgumentError("output_directory must be a string"))
    output = String(output_directory)

    _validate_closed_particles(model, initial, numerics)
    return ClosedRunConfig(model, initial, evolution, numerics, output)
end


function _validate_closed_particles(
        model::ModelConfig,
        initial::ClosedInitialConfig,
        numerics::NumericsConfig,
    )
    if initial.kind == :ground_state
        all(p -> 0 <= p <= n_sites(model), initial.particles) ||
            throw(ArgumentError(
                "ground-state particle counts must lie between 0 and n_sites",
            ))
        numerics.imaginary_time_steps > 0 ||
            throw(ArgumentError(
                "ground-state preparation requires at least one imaginary-time step",
            ))
        return nothing
    end

    regions = (
        (initial.left_particles, model.n_left, "left_particles"),
        (initial.system_particles, n_system(model), "system_particles"),
        (initial.right_particles, model.n_right, "right_particles"),
    )
    for (particles, size, name) in regions
        all(p -> 0 <= p <= size, particles) ||
            throw(ArgumentError("$name must lie between 0 and the region size"))
    end
    return nothing
end


const _MODEL_FIELDS = Set([
    "n_left",
    "n_right",
    "system_shape",
    "lead_basis",
    "lead_grid",
    "log_lambda",
    "lead_hopping",
    "system_hopping",
    "coupling_left",
    "coupling_right",
    "lead_potential_left",
    "lead_potential_right",
    "system_onsite",
    "hubbard_u",
    "density_interaction",
    "interaction_softening",
    "particle_model",
])

const _EVOLUTION_FIELDS = Set(["final_time", "snapshot_interval", "time_step"])

const _NUMERICS_FIELDS = Set([
    "bond_dimension",
    "svd_tolerance",
    "bp_maxiter",
    "bp_tolerance",
    "imaginary_time_steps",
    "imaginary_time_step",
    "ground_state_tolerance",
])

const _INITIAL_FIELDS = Set([
    "kind",
    "particles",
    "left_particles",
    "system_particles",
    "right_particles",
])

const _RUN_FIELDS = Set([
    "model",
    "initial",
    "evolution",
    "numerics",
    "output_directory",
])


function ModelConfig(data)
    values = _config_mapping(data, "ModelConfig")
    _config_reject_unknown(values, _MODEL_FIELDS, "ModelConfig")
    return ModelConfig(
        n_left = _config_required(values, "n_left", "ModelConfig"),
        n_right = _config_required(values, "n_right", "ModelConfig"),
        system_shape = _config_required(values, "system_shape", "ModelConfig"),
        lead_basis = get(values, "lead_basis", :position),
        lead_grid = get(values, "lead_grid", :sine),
        log_lambda = get(values, "log_lambda", nothing),
        lead_hopping = get(values, "lead_hopping", 1.0),
        system_hopping = get(values, "system_hopping", 1.0),
        coupling_left = get(values, "coupling_left", 0.2),
        coupling_right = get(values, "coupling_right", 0.2),
        lead_potential_left = get(values, "lead_potential_left", 0.0),
        lead_potential_right = get(values, "lead_potential_right", 0.0),
        system_onsite = get(values, "system_onsite", 0.0),
        hubbard_u = get(values, "hubbard_u", 0.0),
        density_interaction = get(values, "density_interaction", 0.0),
        interaction_softening = get(values, "interaction_softening", 0.5),
        particle_model = get(values, "particle_model", :hardcore),
    )
end

function EvolutionConfig(data)
    values = _config_mapping(data, "EvolutionConfig")
    _config_reject_unknown(values, _EVOLUTION_FIELDS, "EvolutionConfig")
    return EvolutionConfig(
        final_time = _config_required(values, "final_time", "EvolutionConfig"),
        snapshot_interval = _config_required(
            values,
            "snapshot_interval",
            "EvolutionConfig",
        ),
        time_step = _config_required(values, "time_step", "EvolutionConfig"),
    )
end

function NumericsConfig(data)
    values = _config_mapping(data, "NumericsConfig")
    _config_reject_unknown(values, _NUMERICS_FIELDS, "NumericsConfig")
    return NumericsConfig(
        bond_dimension = get(values, "bond_dimension", 64),
        svd_tolerance = get(values, "svd_tolerance", 1.0e-9),
        bp_maxiter = get(values, "bp_maxiter", 25),
        bp_tolerance = get(values, "bp_tolerance", 1.0e-8),
        imaginary_time_steps = get(values, "imaginary_time_steps", 100),
        imaginary_time_step = get(values, "imaginary_time_step", 0.02),
        ground_state_tolerance = get(values, "ground_state_tolerance", 1.0e-8),
    )
end

function ClosedInitialConfig(data)
    values = _config_mapping(data, "ClosedInitialConfig")
    _config_reject_unknown(values, _INITIAL_FIELDS, "ClosedInitialConfig")
    kind = _config_symbol(get(values, "kind", :ground_state), "kind")
    if kind == :product && haskey(values, "particles")
        throw(ArgumentError("product initialization does not use particles"))
    end
    regional = ("left_particles", "system_particles", "right_particles")
    if kind == :ground_state && any(key -> haskey(values, key), regional)
        throw(ArgumentError("ground-state initialization uses only total particles"))
    end
    return ClosedInitialConfig(
        kind = kind,
        particles = get(values, "particles", nothing),
        left_particles = get(values, "left_particles", nothing),
        system_particles = get(values, "system_particles", nothing),
        right_particles = get(values, "right_particles", nothing),
    )
end

function ClosedRunConfig(data)
    values = _config_mapping(data, "ClosedRunConfig")
    _config_reject_unknown(values, _RUN_FIELDS, "ClosedRunConfig")
    model = ModelConfig(_config_required(values, "model", "ClosedRunConfig"))
    initial = ClosedInitialConfig(_config_required(values, "initial", "ClosedRunConfig"))
    evolution = EvolutionConfig(_config_required(values, "evolution", "ClosedRunConfig"))
    raw_numerics = get(values, "numerics", nothing)
    numerics = isnothing(raw_numerics) ? NumericsConfig() : NumericsConfig(raw_numerics)
    return ClosedRunConfig(
        model = model,
        initial = initial,
        evolution = evolution,
        numerics = numerics,
        output_directory = get(values, "output_directory", "closed_results"),
    )
end


"""Load and strictly validate a closed-run JSON configuration."""
function load_closed_config(path::AbstractString)
    document = open(path, "r") do stream
        JSON3.read(read(stream, String))
    end
    return ClosedRunConfig(document)
end


function config_dict(config::ModelConfig)
    onsite = config.system_onsite isa Tuple ? collect(config.system_onsite) :
        config.system_onsite
    return Dict{String,Any}(
        "n_left" => config.n_left,
        "n_right" => config.n_right,
        "system_shape" => collect(config.system_shape),
        "lead_basis" => String(config.lead_basis),
        "lead_grid" => String(config.lead_grid),
        "log_lambda" => config.log_lambda,
        "lead_hopping" => config.lead_hopping,
        "system_hopping" => config.system_hopping,
        "coupling_left" => config.coupling_left,
        "coupling_right" => config.coupling_right,
        "lead_potential_left" => config.lead_potential_left,
        "lead_potential_right" => config.lead_potential_right,
        "system_onsite" => onsite,
        "hubbard_u" => config.hubbard_u,
        "density_interaction" => config.density_interaction,
        "interaction_softening" => config.interaction_softening,
        "particle_model" => String(config.particle_model),
    )
end

config_dict(config::EvolutionConfig) = Dict{String,Any}(
    "final_time" => config.final_time,
    "snapshot_interval" => config.snapshot_interval,
    "time_step" => config.time_step,
)

config_dict(config::NumericsConfig) = Dict{String,Any}(
    "bond_dimension" => config.bond_dimension,
    "svd_tolerance" => config.svd_tolerance,
    "bp_maxiter" => config.bp_maxiter,
    "bp_tolerance" => config.bp_tolerance,
    "imaginary_time_steps" => config.imaginary_time_steps,
    "imaginary_time_step" => config.imaginary_time_step,
    "ground_state_tolerance" => config.ground_state_tolerance,
)

function config_dict(config::ClosedInitialConfig)
    if config.kind == :ground_state
        return Dict{String,Any}(
            "kind" => "ground_state",
            "particles" => collect(config.particles),
        )
    end
    return Dict{String,Any}(
        "kind" => "product",
        "left_particles" => collect(config.left_particles),
        "system_particles" => collect(config.system_particles),
        "right_particles" => collect(config.right_particles),
    )
end

config_dict(config::ClosedRunConfig) = Dict{String,Any}(
    "model" => config_dict(config.model),
    "initial" => config_dict(config.initial),
    "evolution" => config_dict(config.evolution),
    "numerics" => config_dict(config.numerics),
    "output_directory" => config.output_directory,
)

"""Serialize a configuration to JSON using the same schema accepted by the loader."""
config_json(config) = String(JSON3.write(config_dict(config)))


function _config_mapping(data, context::AbstractString)
    (data isa AbstractDict || data isa NamedTuple) ||
        throw(ArgumentError("$context must be a JSON object"))
    return Dict{String,Any}(String(key) => value for (key, value) in pairs(data))
end

function _config_reject_unknown(
        values::Dict{String,Any},
        allowed::Set{String},
        context::AbstractString,
    )
    unknown = sort!(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) || throw(ArgumentError("unknown $context fields: $unknown"))
    return nothing
end

function _config_required(
        values::Dict{String,Any},
        name::String,
        context::AbstractString,
    )
    haskey(values, name) || throw(ArgumentError("missing required $context field: $name"))
    return values[name]
end

function _config_int(value, name::AbstractString; allow_integral_real::Bool = false)
    value isa Bool && throw(ArgumentError("$name must be an integer"))
    if value isa Integer
        try
            return Int(value)
        catch error
            error isa InexactError || rethrow()
            throw(ArgumentError("$name is outside the supported integer range"))
        end
    end
    if allow_integral_real && value isa Real && isfinite(value) && isinteger(value)
        try
            return Int(value)
        catch error
            error isa InexactError || rethrow()
            throw(ArgumentError("$name is outside the supported integer range"))
        end
    end
    throw(ArgumentError("$name must be an integer"))
end

function _config_int_pair(value, name::AbstractString; allow_integral_reals::Bool = false)
    (value isa Tuple || value isa AbstractVector) && length(value) == 2 ||
        throw(ArgumentError("$name must contain exactly two values"))
    return (
        _config_int(value[1], "$name[1]"; allow_integral_real = allow_integral_reals),
        _config_int(value[2], "$name[2]"; allow_integral_real = allow_integral_reals),
    )
end

function _config_float(value, name::AbstractString)
    value isa Bool && throw(ArgumentError("$name must be a finite real number"))
    value isa Real || throw(ArgumentError("$name must be a finite real number"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$name must be a finite real number"))
    return converted
end

function _config_symbol(value, name::AbstractString)
    (value isa Symbol || value isa AbstractString) ||
        throw(ArgumentError("$name must be a string or Symbol"))
    return Symbol(value)
end

function _config_system_onsite(value)
    if value isa Tuple || value isa AbstractVector
        return Tuple(
            _config_float(entry, "system_onsite[$index]")
            for (index, entry) in enumerate(value)
        )
    end
    return _config_float(value, "system_onsite")
end
