#!/usr/bin/env julia

using BPTransport

function main(args = ARGS)
    if !isempty(args) && first(args) in ("-h", "--help")
        println("usage: julia --project=. examples/run_closed.jl [config.json]")
        return nothing
    end
    length(args) <= 1 || error("expected at most one configuration path")

    default_config = joinpath(@__DIR__, "closed_example.json")
    config_path = isempty(args) ? default_config : only(args)
    config = load_closed_config(config_path)
    result = run_closed(config)
    data = result.data

    println("Completed closed BP transport run with $(length(data["time"])) snapshots.")
    println("Final time: $(data["time"][end])")
    println("Final norm: $(data["norm"][end])")
    println("Final up occupations: $(data["occupation_up"][end, :])")
    println("Final down occupations: $(data["occupation_down"][end, :])")
    println("Final up currents [left -> system, system -> right]: $(data["current_up"][end, :])")
    println("Observables: $(joinpath(config.output_directory, "observables.h5"))")
    println("Checkpoint: $(joinpath(config.output_directory, "final_state.jls"))")
    return result
end

main()
