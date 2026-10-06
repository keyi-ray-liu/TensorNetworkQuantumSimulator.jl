"""Shared time-grid helpers for BP transport evolution."""

"""
    snapshot_times(config::EvolutionConfig) -> Vector{Float64}

Return a monotonically increasing snapshot grid containing both zero and the
exact requested final time. The last interval may be shorter than
`snapshot_interval`.
"""
function snapshot_times(config::EvolutionConfig)
    final_time = config.final_time
    final_time == 0 && return [0.0]

    count = floor(Int, final_time / config.snapshot_interval)
    times = config.snapshot_interval .* collect(0:count)

    if length(times) == 1
        return [0.0, final_time]
    end

    if isapprox(last(times), final_time; rtol = 0.0, atol = 1.0e-14)
        times[end] = final_time
    else
        push!(times, final_time)
    end
    return times
end
