# BPTransport

`BPTransport` is a small closed-system transport package built on
[`TensorNetworkQuantumSimulator.jl`](../README.md). It follows the layout and
observable conventions of the transport examples in `yastn-transport`, while
using belief-propagation (BP) environments and local simple-update gates.

> [!IMPORTANT]
> This package simulates **hard-core modes represented by spin-1/2 tensors**.
> It is not a general fermionic tensor-network implementation. The
> `:fermion_1d` interpretation is exact only in the restricted one-dimensional
> setting described below, where reported one-body correlators include explicit
> Jordan--Wigner strings. The package currently supports closed, unitary
> evolution only; it does not implement Lindblad reservoirs or an open-system
> steady-state solver.

## Physical model and conventions

Every physical site has independent `:up` and `:down` hard-core modes. For each
mode, `|↑⟩` is empty, `|↓⟩` is occupied, and

```math
n = \frac{1-Z}{2}.
```

The Hamiltonian contains onsite potentials, flavor-preserving hopping, onsite
up-down interactions, and optional device density interactions. A hopping term
is

```math
h_{ij} b_i^\dagger b_j + h_{ij}^* b_j^\dagger b_i,
```

and its oriented current is reported using

```math
J(i\rightarrow j) = -2\,\operatorname{Im}
\left[h_{ij}\langle b_i^\dagger b_j\rangle\right].
```

Stored contact currents are ordered as `[left -> system, system -> right]`.
Physical sites use the labels `L(k)`, `S(k)`, and `R(k)`. The device uses
row-major ordering for `system_shape = [rows, columns]`.

The model selector has two meanings:

- `particle_model = "hardcore"` is the native interpretation. Each flavor is a
  distinguishable hard-core species, and the model is valid on every supported
  geometry.
- `particle_model = "fermion_1d"` uses the hard-core/Jordan-Wigner equivalence.
  It is exact only for position-basis, strictly one-dimensional geometries with
  nearest-neighbor flavor-preserving hopping, density-only interactions, and
  no spin mixing or exchange. Its correlators use the declared global order
  `up(Ln,...,L1,S1,...,SN,R1,...,Rn)` followed by the same down-mode order,
  including the required parity strings. Mixed lead modes, two-dimensional
  devices, and periodic hopping are outside this interpretation.

## Numerical scope

There are three independent approximation controls:

1. BP contracts tree tensor networks exactly but is approximate on loopy
   graphs. `bp_maxiter` and `bp_tolerance` control message convergence. A
   failure to converge is currently reported as a runtime warning rather than
   stored as a result field, so warnings from production runs should be kept.
2. Two-site gates use simple update. It is exact when no singular values are
   discarded; `bond_dimension` and `svd_tolerance` control truncation.
3. Time evolution uses a palindromic second-order Suzuki--Trotter circuit.
   Decreasing `time_step` controls the Trotter error. The last internal step is
   shortened so the requested snapshot and final times are reached exactly.

Ground-state preparation, when selected, uses finite imaginary-time evolution
and therefore has the same BP, truncation, and finite-step qualifications. Its
`converged` diagnostic means only that three consecutive energy changes met
`ground_state_tolerance`; it is an energy-stationarity heuristic, not an
eigenstate certificate. Product-state preparation is deterministic.

The tensors do not currently carry U(1) quantum-number blocks. Evolution gates
conserve each flavor before truncation, but a truncated simple update does not
mathematically guarantee an exact fixed-number sector. The recorded
`total_up`/`total_down` histories make any resulting drift visible.

Mixed-basis leads are represented as literal stars, and long-range density
terms become literal graph edges. Tensor cost grows rapidly with vertex degree,
so those options are intended for small lead/device counts; position-basis
chains are the scalable default.

## Installation

From this directory, develop the parent checkout and instantiate the transport
environment:

```bash
cd TRANSPORT
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

This deliberately uses the local parent package, so changes to
`TensorNetworkQuantumSimulator.jl` are immediately visible to `BPTransport`.

## Run the example

```bash
julia --project=. examples/run_closed.jl examples/closed_example.json
```

With no argument, the driver uses `examples/closed_example.json`. The example is
a minimal `L1-S1-R1` position-basis device, initially containing one up particle
in the left lead. Its requested times are `0.0`, `0.02`, and `0.035`, exercising
an internal final step shorter than `time_step`.

The same workflow is available from Julia:

```julia
using BPTransport

config = load_closed_config("examples/closed_example.json")
result = run_closed(config)

result.data["time"]
result.data["occupation_up"]
result.data["current_up"]
```

`run_closed` returns a `TransportResult`. Its `cache` is the final updated BP
cache, `model` is the built transport model, and `data` is a dictionary of
time-major observable arrays. Passing `result.cache` and `result.model` to
`measure_transport` performs another measurement without rebuilding the BP
environment.

## Configuration

Configuration files are strict JSON: unknown fields are rejected. The top-level
sections are:

- `model`: lead sizes, `system_shape`, position or mixed lead basis, hoppings,
  contact couplings, potentials, interactions, and `particle_model`.
- `initial`: either `kind = "product"` with two-component regional particle
  counts `[up, down]`, or `kind = "ground_state"` with total `particles`.
- `evolution`: `final_time`, `snapshot_interval`, and internal `time_step`.
- `numerics`: bond dimension, SVD cutoff, BP convergence controls, and
  imaginary-time preparation controls.
- `output_directory`: destination for a fresh run's result and checkpoint.

For position-basis leads, `L(1)` and `R(1)` are adjacent to the device. Mixed
leads diagonalize the lead Hamiltonians and connect each lead mode to its device
contact. See [`examples/closed_example.json`](examples/closed_example.json) for
a complete product-state configuration.

## Results and checkpoints

Each run returns the observables in memory and writes:

- `observables.h5`: exact snapshot times, physical and mode labels,
  spin-resolved occupations and currents, particle totals, norm, energy, bond
  dimension maximum, device correlation matrices, the full JSON configuration,
  and observable conventions.
- `final_state.jls`: the final tensor-network state plus configuration,
  accumulated data, and preparation metadata.

Load a checkpoint with:

```julia
restored = load_checkpoint("closed_results")
final_values = measure_transport(restored.cache, restored.model)
```

The checkpoint uses Julia `Serialization`, must be treated as trusted input,
and is intended for the same Julia and package environment rather than as a
long-term cross-version archive. The HDF5 observable file is the portable
analysis output. A fresh run accumulates snapshots in memory and commits these
two files at completion; interrupted-run resume is not yet implemented.
Concurrent runs should use distinct `output_directory` values.

## Tests

After installation:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The focused suite covers strict configuration parsing, exact snapshot
endpoints, position and mixed lead formulas, graph locality, interaction-energy
factors, product and imaginary-time preparation, analytic current flow,
Jordan--Wigner signs, particle and norm behavior in untruncated fixtures,
output shapes, and checkpoint restoration.
