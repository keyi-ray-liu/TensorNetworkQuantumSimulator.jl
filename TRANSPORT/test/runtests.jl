using BPTransport
using HDF5
using TensorNetworkQuantumSimulator: has_edge, network, norm_sqr, tensornetworkstate
using Test

const EXAMPLE_CONFIG = normpath(
    joinpath(@__DIR__, "..", "examples", "closed_example.json"),
)

function fixture_json(
        output_directory;
        coupling_left = 0.2,
        coupling_right = 0.35,
        final_time = 0.035,
        snapshot_interval = 0.02,
        time_step = 0.01,
    )
    return """
    {
      "model": {
        "n_left": 1,
        "n_right": 1,
        "system_shape": [1, 1],
        "lead_basis": "position",
        "lead_grid": "sine",
        "lead_hopping": 0.0,
        "system_hopping": 0.0,
        "coupling_left": $coupling_left,
        "coupling_right": $coupling_right,
        "lead_potential_left": 0.0,
        "lead_potential_right": 0.0,
        "system_onsite": 0.0,
        "hubbard_u": 0.0,
        "density_interaction": 0.0,
        "interaction_softening": 0.5,
        "particle_model": "hardcore"
      },
      "initial": {
        "kind": "product",
        "left_particles": [1, 0],
        "system_particles": [0, 0],
        "right_particles": [0, 0]
      },
      "evolution": {
        "final_time": $final_time,
        "snapshot_interval": $snapshot_interval,
        "time_step": $time_step
      },
      "numerics": {
        "bond_dimension": 8,
        "svd_tolerance": 1.0e-12,
        "bp_maxiter": 25,
        "bp_tolerance": 1.0e-10,
        "imaginary_time_steps": 8,
        "imaginary_time_step": 0.02,
        "ground_state_tolerance": 1.0e-8
      },
      "output_directory": $(repr(String(output_directory)))
    }
    """
end

function load_fixture(directory; kwargs...)
    config_path = joinpath(directory, "config.json")
    open(config_path, "w") do stream
        write(stream, fixture_json(joinpath(directory, "results"); kwargs...))
    end
    return load_closed_config(config_path)
end

@testset "BPTransport" begin
    @testset "configuration and snapshot grid" begin
        config = load_closed_config(EXAMPLE_CONFIG)
        @test config.model.n_left == 1
        @test config.model.n_right == 1
        @test config.model.system_shape == (1, 1)
        @test config.model.particle_model == :hardcore
        @test config.initial.kind == :product
        @test config.initial.left_particles == (1, 0)

        times = snapshot_times(config.evolution)
        @test times ≈ [0.0, 0.02, 0.035]
        @test isequal(times[end], config.evolution.final_time)

        irregular = EvolutionConfig(
            final_time = 0.1,
            snapshot_interval = 0.03,
            time_step = 0.02,
        )
        irregular_times = snapshot_times(irregular)
        @test irregular_times ≈ [0.0, 0.03, 0.06, 0.09, 0.1]
        @test isequal(irregular_times[end], irregular.final_time)
        @test snapshot_times(
            EvolutionConfig(
                final_time = 0.0,
                snapshot_interval = 1.0,
                time_step = 0.1,
            ),
        ) == [0.0]

        @test_throws ArgumentError EvolutionConfig(
            final_time = -0.1,
            snapshot_interval = 0.1,
            time_step = 0.01,
        )
        @test_throws ArgumentError ModelConfig(
            n_left = 0,
            n_right = 1,
            system_shape = (1, 1),
        )
        @test_throws ArgumentError ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (2, 2),
            particle_model = :fermion_1d,
        )

        mktempdir() do directory
            document = replace(
                read(EXAMPLE_CONFIG, String),
                "{\n" => "{\n  \"unexpected\": true,\n";
                count = 1,
            )
            bad_path = joinpath(directory, "unknown-field.json")
            open(bad_path, "w") do stream
                write(stream, document)
            end
            @test_throws ArgumentError load_closed_config(bad_path)
        end
    end

    @testset "minimal model structure" begin
        config = load_closed_config(EXAMPLE_CONFIG)
        model = build_model(config.model)

        @test model.config == config.model
        @test model.sites == (L(1), S(1), R(1))
        @test model.sites == canonical_sites(config.model)
        @test model.vertices == [
            mode_vertex(site, flavor)
            for site in model.sites for flavor in (:up, :down)
        ]
        @test length(model.contacts) == 2
        @test getfield.(model.contacts, :side) == (:left, :right)
        @test first(model.contacts).origin == L(1)
        @test first(model.contacts).destination == S(1)
        @test last(model.contacts).origin == S(1)
        @test last(model.contacts).destination == R(1)
    end

    @testset "mixed sine and logarithmic leads" begin
        sine_config = ModelConfig(
            n_left = 2,
            n_right = 3,
            system_shape = (1, 2),
            lead_basis = :mixed,
            lead_grid = :sine,
            lead_hopping = 1.25,
            system_hopping = 0.5,
            coupling_left = 0.6,
            coupling_right = 0.8,
            lead_potential_left = -0.3,
            lead_potential_right = 0.4,
            system_onsite = (0.1, -0.2),
        )
        sine_model = build_model(sine_config)
        left_first = lead_mode(sine_config, :left, 1)
        right_middle = lead_mode(sine_config, :right, 2)

        @test length(sine_model.lead_modes) == 5
        @test length(sine_model.contacts) == 5
        @test collect(getfield.(sine_model.contacts, :amplitude)) ≈
            ComplexF64.(getfield.(sine_model.lead_modes, :contact))
        @test left_first.energy ≈ -0.3 + 2 * 1.25 * cos(π / 3)
        @test left_first.contact ≈ 0.6 * sin(π / 3) * sqrt(2 / 3)
        @test right_middle.energy ≈ 0.4 + 2 * 1.25 * cos(π / 2)
        @test right_middle.contact ≈ 0.8 * sin(π / 2) * sqrt(2 / 4)
        @test count(term -> term.kind == :hopping, sine_model.bond_terms) == 12
        @test length(sine_model.dummy_edges) == 1

        log_config = ModelConfig(
            n_left = 2,
            n_right = 4,
            system_shape = (1, 1),
            lead_basis = :mixed,
            lead_grid = :log,
            log_lambda = 2.0,
            lead_hopping = 1.5,
            system_hopping = 0.0,
            coupling_left = 0.6,
            coupling_right = 0.8,
            lead_potential_left = -0.3,
            lead_potential_right = 0.4,
        )
        log_model = build_model(log_config)
        left_modes = filter(mode -> mode.side == :left, log_model.lead_modes)
        right_modes = filter(mode -> mode.side == :right, log_model.lead_modes)

        @test length(log_model.lead_modes) == 6
        @test length(log_model.contacts) == 6
        @test left_modes[1].energy - (-0.3) ≈ -(left_modes[2].energy - (-0.3))
        @test left_modes[1].contact ≈ left_modes[2].contact
        @test right_modes[1].energy - 0.4 ≈ -(right_modes[4].energy - 0.4)
        @test right_modes[2].energy - 0.4 ≈ -(right_modes[3].energy - 0.4)
        @test right_modes[1].contact ≈ right_modes[4].contact
        @test right_modes[2].contact ≈ right_modes[3].contact
        @test count(term -> term.kind == :hopping, log_model.bond_terms) == 12
        @test length(log_model.dummy_edges) == 1
    end

    @testset "dummy and Hamiltonian-term graph invariants" begin
        structural_model = build_model(ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 1),
            lead_hopping = 0.0,
            system_hopping = 0.0,
            coupling_left = 0.0,
            coupling_right = 0.0,
        ))
        dummy = only(structural_model.dummy_edges)

        @test isempty(structural_model.bond_terms)
        @test dummy.src == mode_vertex(S(1), :up)
        @test dummy.dst == mode_vertex(S(1), :down)
        @test has_edge(structural_model.graph, dummy)
        for flavor in (:up, :down)
            @test has_edge(
                structural_model.graph,
                mode_vertex(L(1), flavor),
                mode_vertex(S(1), flavor),
            )
            @test has_edge(
                structural_model.graph,
                mode_vertex(S(1), flavor),
                mode_vertex(R(1), flavor),
            )
        end

        interacting_model = build_model(ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 2),
            lead_hopping = 0.0,
            system_hopping = 0.7,
            coupling_left = 0.2,
            coupling_right = 0.3,
            hubbard_u = 1.1,
            density_interaction = 0.9,
            interaction_softening = 0.5,
        ))

        @test isempty(interacting_model.dummy_edges)
        @test count(term -> term.kind == :hopping, interacting_model.bond_terms) == 6
        @test count(term -> term.kind == :hubbard, interacting_model.bond_terms) == 2
        @test count(term -> term.kind == :density, interacting_model.bond_terms) == 4
        @test all(
            term -> has_edge(
                interacting_model.graph,
                term.origin,
                term.destination,
            ),
            interacting_model.bond_terms,
        )
    end

    @testset "onsite, Hubbard, and density energy factors" begin
        model_config = ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 2),
            lead_hopping = 0.0,
            system_hopping = 0.0,
            coupling_left = 0.0,
            coupling_right = 0.0,
            lead_potential_left = 0.25,
            lead_potential_right = -0.75,
            system_onsite = (2.0, -1.5),
            hubbard_u = 3.0,
            density_interaction = 4.0,
            interaction_softening = 0.5,
        )
        model = build_model(model_config)
        hubbard_terms = filter(term -> term.kind == :hubbard, model.bond_terms)
        density_terms = filter(term -> term.kind == :density, model.bond_terms)

        @test length(model.onsite_terms) == 8
        @test length(hubbard_terms) == 2
        @test all(term -> term.amplitude ≈ 3.0, hubbard_terms)
        @test length(density_terms) == 4
        @test all(term -> term.amplitude ≈ 4 / (1 + 0.5), density_terms)

        run_config = ClosedRunConfig(
            model = model_config,
            initial = ClosedInitialConfig(
                kind = :product,
                left_particles = (1, 0),
                system_particles = (2, 1),
                right_particles = (0, 1),
            ),
            evolution = EvolutionConfig(
                final_time = 0.0,
                snapshot_interval = 1.0,
                time_step = 0.1,
            ),
            numerics = NumericsConfig(bond_dimension = 4),
        )
        cache, _ = prepare_closed_state(run_config, model)
        values = measure_transport(cache, model; include_correlations = false)
        onsite_energy = 0.25 + 2.0 - 1.5 + 2.0 - 0.75
        hubbard_energy = 3.0
        density_energy = 2 * 4 / (1 + 0.5)

        @test values["occupation_up"] ≈ [1.0, 1.0, 1.0, 0.0] atol = 1.0e-12
        @test values["occupation_down"] ≈ [0.0, 1.0, 0.0, 1.0] atol = 1.0e-12
        @test values["energy"] ≈
            onsite_energy + hubbard_energy + density_energy atol = 1.0e-10
    end

    @testset "fermion_1d Jordan-Wigner correlator" begin
        fermion_config = ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 3),
            lead_hopping = 0.0,
            system_hopping = 0.0,
            coupling_left = 0.0,
            coupling_right = 0.0,
            particle_model = :fermion_1d,
        )
        fermion_model = build_model(fermion_config)
        middle = mode_vertex(S(2), :up)
        endpoints = (mode_vertex(S(1), :up), mode_vertex(S(3), :up))
        state = tensornetworkstate(
            ComplexF64,
            vertex -> vertex == middle ? "↓" : vertex in endpoints ? "X+" : "↑",
            fermion_model.graph,
            "S=1/2",
        )
        fermion_values = measure_transport(state, fermion_model; alg = "exact")

        hardcore_model = build_model(ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 3),
            lead_hopping = 0.0,
            system_hopping = 0.0,
            coupling_left = 0.0,
            coupling_right = 0.0,
            particle_model = :hardcore,
        ))
        hardcore_values = measure_transport(state, hardcore_model; alg = "exact")

        @test fermion_values["system_cdag_c_up"][1, 3] ≈ -0.25 atol = 1.0e-12
        @test fermion_values["system_cdag_c_up"][3, 1] ≈ -0.25 atol = 1.0e-12
        @test hardcore_values["system_cdag_c_up"][1, 3] ≈ 0.25 atol = 1.0e-12
        @test fermion_values["occupation_up"] ≈
            [0.0, 0.5, 1.0, 0.5, 0.0] atol = 1.0e-12
    end

    @testset "product preparation" begin
        config = load_closed_config(EXAMPLE_CONFIG)
        model = build_model(config.model)
        cache, preparation = prepare_closed_state(config, model)
        values = measure_transport(cache, model; include_correlations = false)

        @test preparation["kind"] == "product"
        @test preparation["converged"]
        @test values["occupation_up"] ≈ [1.0, 0.0, 0.0] atol = 1.0e-12
        @test values["occupation_down"] ≈ zeros(3) atol = 1.0e-12
        @test values["current_up"] ≈ zeros(2) atol = 1.0e-12
        @test values["current_down"] ≈ zeros(2) atol = 1.0e-12
        @test values["total_up"] ≈ 1.0 atol = 1.0e-12
        @test values["total_down"] ≈ 0.0 atol = 1.0e-12
        @test values["norm"] ≈ 1.0 atol = 1.0e-12
        @test real(norm_sqr(network(cache); alg = "exact")) ≈ 1.0 atol = 1.0e-12
    end

    @testset "imaginary-time ground-state preparation" begin
        model_config = ModelConfig(
            n_left = 1,
            n_right = 1,
            system_shape = (1, 1),
            lead_hopping = 0.0,
            system_hopping = 0.0,
            coupling_left = 0.0,
            coupling_right = 0.0,
            lead_potential_left = 1.0,
            lead_potential_right = 2.0,
            system_onsite = -1.0,
        )
        config = ClosedRunConfig(
            model = model_config,
            initial = ClosedInitialConfig(
                kind = :ground_state,
                particles = (1, 0),
            ),
            evolution = EvolutionConfig(
                final_time = 0.0,
                snapshot_interval = 1.0,
                time_step = 0.1,
            ),
            numerics = NumericsConfig(
                bond_dimension = 4,
                bp_maxiter = 10,
                bp_tolerance = 1.0e-10,
                imaginary_time_steps = 3,
                imaginary_time_step = 0.1,
                ground_state_tolerance = 1.0e-8,
            ),
        )
        model = build_model(model_config)
        cache, preparation = prepare_closed_state(config, model)
        values = measure_transport(cache, model; include_correlations = false)

        @test preparation["kind"] == "ground_state"
        @test preparation["converged"]
        @test preparation["imaginary_time_steps"] == 3
        @test preparation["convergence_criterion"] ==
            "three consecutive |delta energy| <= ground_state_tolerance"
        @test preparation["last_energy_change"] <=
            config.numerics.ground_state_tolerance
        @test preparation["energy"] ≈ -1.0 atol = 1.0e-10
        @test values["occupation_up"] ≈ [0.0, 1.0, 0.0] atol = 1.0e-10
        @test values["occupation_down"] ≈ zeros(3) atol = 1.0e-10
        @test values["total_up"] ≈ 1.0 atol = 1.0e-10
        @test values["norm"] ≈ 1.0 atol = 1.0e-10
        @test real(norm_sqr(network(cache); alg = "exact")) ≈ 1.0 atol = 1.0e-10
    end

    @testset "analytic one-contact transfer and current" begin
        mktempdir() do directory
            hopping = 0.7
            duration = 0.1
            config = load_fixture(
                directory;
                coupling_left = hopping,
                coupling_right = 0.0,
                final_time = duration,
                snapshot_interval = duration,
                time_step = duration,
            )
            model = build_model(config.model)
            cache, _ = prepare_closed_state(config, model)
            cache, maximum_error, _ = evolve_interval!(
                cache,
                model,
                duration;
                time_step = duration,
                numerics = config.numerics,
            )
            values = measure_transport(cache, model; include_correlations = false)

            angle = hopping * duration
            expected_occupations = [cos(angle)^2, sin(angle)^2, 0.0]
            expected_current = hopping * sin(2angle)
            @test values["occupation_up"] ≈ expected_occupations atol = 1.0e-10
            @test values["occupation_down"] ≈ zeros(3) atol = 1.0e-12
            @test values["current_up"] ≈ [expected_current, 0.0] atol = 1.0e-10
            @test values["current_down"] ≈ zeros(2) atol = 1.0e-12
            @test sum(values["occupation_up"]) ≈ 1.0 atol = 1.0e-10
            @test values["norm"] ≈ 1.0 atol = 1.0e-10
            @test maximum_error ≤ 1.0e-10
        end
    end

    @testset "stationary zero-coupling dynamics" begin
        mktempdir() do directory
            config = load_fixture(
                directory;
                coupling_left = 0.0,
                coupling_right = 0.0,
                final_time = 0.02,
                snapshot_interval = 0.01,
                time_step = 0.007,
            )
            result = run_closed(config; write_output = false, verbose = false)
            data = result.data

            @test data["time"] ≈ [0.0, 0.01, 0.02]
            @test data["occupation_up"] ≈ repeat(
                reshape([1.0, 0.0, 0.0], 1, :),
                3,
                1,
            ) atol = 1.0e-12
            @test data["occupation_down"] ≈ zeros(3, 3) atol = 1.0e-12
            @test data["current_up"] ≈ zeros(3, 2) atol = 1.0e-12
            @test data["current_down"] ≈ zeros(3, 2) atol = 1.0e-12
            @test data["total_up"] ≈ ones(3) atol = 1.0e-12
            @test data["total_down"] ≈ zeros(3) atol = 1.0e-12
            @test data["norm"] ≈ ones(3) atol = 1.0e-12
            @test data["max_truncation_error"] ≈ zeros(3) atol = 1.0e-12
        end
    end

    @testset "position-basis end-to-end run and output" begin
        mktempdir() do directory
            config = load_fixture(directory)
            result = run_closed(config; write_output = true, verbose = false)
            data = result.data
            required = Set([
                "time",
                "occupation_up",
                "occupation_down",
                "current_up",
                "current_down",
                "total_up",
                "total_down",
                "norm",
                "energy",
                "max_bond_dimension",
                "max_truncation_error",
                "system_cdag_c_up",
                "system_cdag_c_down",
                "system_nn_up_down",
            ])
            @test issubset(required, Set(keys(data)))

            @test data["time"] ≈ [0.0, 0.02, 0.035]
            @test isequal(data["time"][end], config.evolution.final_time)
            @test size(data["occupation_up"]) == (3, 3)
            @test size(data["occupation_down"]) == (3, 3)
            @test size(data["current_up"]) == (3, 2)
            @test size(data["current_down"]) == (3, 2)
            @test size(data["system_cdag_c_up"]) == (3, 1, 1)
            @test size(data["system_nn_up_down"]) == (3, 1, 1)

            @test data["occupation_up"][1, :] ≈ [1.0, 0.0, 0.0] atol = 1.0e-12
            @test data["occupation_down"][1, :] ≈ zeros(3) atol = 1.0e-12
            @test vec(sum(data["occupation_up"]; dims = 2)) ≈ ones(3) atol = 1.0e-8
            @test vec(sum(data["occupation_down"]; dims = 2)) ≈ zeros(3) atol = 1.0e-8
            @test data["total_up"] ≈ ones(3) atol = 1.0e-8
            @test data["total_down"] ≈ zeros(3) atol = 1.0e-8
            @test data["norm"] ≈ ones(3) atol = 1.0e-8
            @test all(isfinite, data["energy"])
            @test all(isfinite, data["current_up"])
            @test all(isfinite, data["current_down"])
            @test all(data["max_bond_dimension"] .<= config.numerics.bond_dimension)
            @test all(data["max_truncation_error"] .>= 0)

            final_values = measure_transport(
                result.cache,
                result.model;
                include_correlations = false,
            )
            exact_values = measure_transport(
                network(result.cache),
                result.model;
                alg = "exact",
                include_correlations = false,
            )
            for name in ("occupation_up", "occupation_down", "current_up", "current_down")
                @test final_values[name] ≈ data[name][end, :] atol = 1.0e-10
                @test final_values[name] ≈ exact_values[name] atol = 1.0e-8
            end
            @test final_values["norm"] ≈ exact_values["norm"] atol = 1.0e-8
            @test final_values["energy"] ≈ exact_values["energy"] atol = 1.0e-8

            observable_path = joinpath(config.output_directory, "observables.h5")
            checkpoint_path = joinpath(config.output_directory, "final_state.jls")
            @test isfile(observable_path)
            @test isfile(checkpoint_path)
            h5open(observable_path, "r") do file
                @test read(file["time"]) ≈ data["time"]
                @test size(file["occupation_up"]) == (3, 3)
                @test size(file["current_up"]) == (3, 2)
                @test size(file["physical_sites"]) == (3,)
                @test size(file["mode_vertices"]) == (6,)
            end

            restored = load_checkpoint(config.output_directory)
            @test restored.data["time"] == data["time"]
            restored_values = measure_transport(
                restored.cache,
                restored.model;
                include_correlations = false,
            )
            @test restored_values["occupation_up"] ≈ final_values["occupation_up"] atol = 1.0e-10
            @test restored_values["current_up"] ≈ final_values["current_up"] atol = 1.0e-10
            @test restored_values["norm"] ≈ final_values["norm"] atol = 1.0e-10
        end
    end
end
