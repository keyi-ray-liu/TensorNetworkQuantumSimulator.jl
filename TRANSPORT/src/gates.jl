const _Term = Union{OnsiteTerm, BondTerm}

_number_operator(site_index) = (op("I", site_index) - op("Z", site_index)) / 2

function _term_hamiltonian(term::OnsiteTerm, state)
    site_index = only(siteinds(state, term.vertex))
    return term.amplitude * _number_operator(site_index)
end

function _term_hamiltonian(term::BondTerm, state)
    source_index = only(siteinds(state, term.origin))
    destination_index = only(siteinds(state, term.destination))

    if term.kind == :hopping
        forward = op("S-", source_index) * op("S+", destination_index)
        backward = op("S+", source_index) * op("S-", destination_index)
        return term.amplitude * forward + conj(term.amplitude) * backward
    elseif term.kind in (:hubbard, :density)
        return term.amplitude *
            _number_operator(source_index) * _number_operator(destination_index)
    end

    throw(ArgumentError("unsupported bond-term kind $(repr(term.kind))"))
end

_term_vertices(term::OnsiteTerm) = [term.vertex]
_term_vertices(term::BondTerm) = [term.origin, term.destination]

function _term_gate(term::_Term, state, step::Real; imaginary::Bool)
    generator = imaginary ? -step : -im * step
    return exp(generator * _term_hamiltonian(term, state))
end

"""
    trotter_circuit(model, state, step; imaginary=false)

Build one palindromic second-order Suzuki--Trotter step. Each nonzero local
Hamiltonian term appears for a half step in forward order and for a half step
in reverse order. The returned named tuple contains raw `ITensor` gates and
their graph vertices, ready for `apply_gates`.
"""
function trotter_circuit(
        model::BuiltModel,
        state,
        step::Real;
        imaginary::Bool = false,
    )
    step > 0 || throw(ArgumentError("Trotter step must be positive"))
    terms = _Term[]
    append!(terms, filter(term -> !iszero(term.amplitude), model.onsite_terms))
    append!(terms, filter(term -> !iszero(term.amplitude), model.bond_terms))

    gates = ITensor[]
    gate_vertices = Any[]
    half_step = step / 2
    for term in Iterators.flatten((terms, Iterators.reverse(terms)))
        push!(gates, _term_gate(term, state, half_step; imaginary))
        push!(gate_vertices, _term_vertices(term))
    end
    return (; gates, gate_vertices)
end

function _bp_update_kwargs(numerics::NumericsConfig; verbose::Bool = false)
    return (
        maxiter = numerics.bp_maxiter,
        tolerance = numerics.bp_tolerance,
        verbose = verbose,
    )
end

"""
    evolve_interval!(cache, model, duration; time_step, numerics,
                     imaginary=false, verbose=false)

Advance a BP cache over exactly `duration`, shortening the last internal step
when necessary. `BeliefPropagationCache` is immutable and `apply_gates` returns
a copy, so callers must use the returned cache despite the conventional bang
in this routine's name.
"""
function evolve_interval!(
        cache::BeliefPropagationCache,
        model::BuiltModel,
        duration::Real;
        time_step::Real,
        numerics::NumericsConfig,
        imaginary::Bool = false,
        verbose::Bool = false,
    )
    duration >= 0 || throw(ArgumentError("duration must be non-negative"))
    time_step > 0 || throw(ArgumentError("time_step must be positive"))

    all_errors = Float64[]
    elapsed = 0.0
    target = float(duration)
    while elapsed < target
        step = min(float(time_step), target - elapsed)
        circuit = trotter_circuit(model, cache, step; imaginary)
        if !isempty(circuit.gates)
            cache, errors = apply_gates(
                circuit.gates,
                cache;
                gate_vertices = circuit.gate_vertices,
                apply_kwargs = (
                    maxdim = numerics.bond_dimension,
                    cutoff = numerics.svd_tolerance,
                    normalize_tensors = false,
                ),
                bp_update_kwargs = _bp_update_kwargs(numerics; verbose),
                verbose = false,
            )
            append!(all_errors, Float64.(real.(errors)))
            imaginary && rescale!(cache)
        end
        next_elapsed = elapsed + step
        next_elapsed > elapsed || error("time stepping made no floating-point progress")
        elapsed = min(target, next_elapsed)
    end

    max_error = isempty(all_errors) ? 0.0 : maximum(all_errors)
    return cache, max_error, all_errors
end

function evolve_interval!(
        cache::BeliefPropagationCache,
        model::BuiltModel,
        duration::Real,
        time_step::Real,
        numerics::NumericsConfig;
        kwargs...,
    )
    return evolve_interval!(cache, model, duration; time_step, numerics, kwargs...)
end
