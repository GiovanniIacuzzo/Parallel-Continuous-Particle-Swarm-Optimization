#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "cpso/topology.hpp"

namespace cpso {

// Objective callbacks used with master--slave, particle-parallel, asynchronous,
// or island execution must be thread-safe. Deterministic objectives are needed
// for exact synchronous equivalence and reproducible paired experiments.
using Objective = std::function<double(const std::vector<double>&)>;

enum class MigrationTopology : int {
    Ring = 0,
    FullyConnected = 1,
};

struct Config {
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::size_t population_size{40};
    std::size_t iterations{100};
    std::uint64_t seed{1};

    double mu_max{0.9};
    double mu_min{0.4};
    double cognitive{2.05};
    double social{2.05};
    double dt{0.7};
    double initial_velocity_fraction{0.1};
    double boundary_restitution{0.5};

    TopologyKind topology{TopologyKind::GlobalBest};
    std::size_t grid_rows{0};
    std::size_t grid_columns{0};
    // Shared structural parameters for graph-based topologies:
    // random: target expected degree; small-world: ring degree (even);
    // scale-free: 2*m preferential-attachment degree parameter.
    std::size_t topology_degree{4};
    double topology_rewiring_probability{0.15};
};

struct IslandConfig {
    std::size_t island_count{4};
    std::size_t worker_count{4};
    // Zero disables migration. Otherwise migration is attempted after every
    // migration_interval completed synchronous island generations.
    std::size_t migration_interval{0};
    MigrationTopology migration_topology{MigrationTopology::Ring};
};

struct Result {
    double best_value{};
    std::vector<double> best_position;
    // Synchronous modes: initial point, then one point per generation.
    // Asynchronous mode: initial point, then one point per committed particle update.
    std::vector<double> best_history;
    // Number of objective evaluations completed when the corresponding history
    // value was observed. This permits fair FE-axis comparisons across modes.
    std::vector<std::uint64_t> history_evaluations;
    std::uint64_t function_evaluations{};
    std::size_t iterations_completed{};
    double elapsed_seconds{};
    // Wall time across synchronous objective batches (initialization + generations).
    // For asynchronous updates, objective-call work is instead exposed via
    // evaluation_worker_seconds and async_update_wall_seconds.
    double evaluation_wall_seconds{};
    // Wall time across synchronous propagation phases.
    double propagation_seconds{};
    double topology_setup_seconds{};
    double social_selection_seconds{};
    double initialization_seconds{};
    // Sum of per-task compute times; can exceed elapsed wall time under concurrency.
    double propagation_worker_seconds{};
    double evaluation_worker_seconds{};
    double migration_seconds{};
    double async_update_wall_seconds{};
    std::uint64_t migrations_performed{};
    std::size_t worker_count{1};
    std::size_t island_count{1};
};

void validate_config(const Config& config);
[[nodiscard]] Result minimize_serial(const Config& config, const Objective& objective);

// Only objective evaluations are dispatched concurrently. Propagation,
// random-number generation and state commits remain serial.
[[nodiscard]] Result minimize_master_slave(
    const Config& config,
    const Objective& objective,
    std::size_t worker_count);

// Particle-parallel synchronous CPSO: particle initialization and propagation
// are parallelized, followed by a parallel objective batch. Social information
// is frozen at each generation boundary, so this mode must match serial exactly.
[[nodiscard]] Result minimize_particle_parallel(
    const Config& config,
    const Objective& objective,
    std::size_t worker_count);

// Event-driven asynchronous global/ring/grid CPSO. Each particle advances its
// own iteration counter and publishes its personal best immediately after an
// objective evaluation. Completion order is schedule-dependent when workers>1.
[[nodiscard]] Result minimize_asynchronous(
    const Config& config,
    const Objective& objective,
    std::size_t worker_count);

// Coarse-grained island model. The total population is divided equally among
// islands; each island is advanced serially while islands execute concurrently.
// Ring or fully-connected best-to-worst migration is applied synchronously.
[[nodiscard]] Result minimize_islands(
    const Config& config,
    const Objective& objective,
    const IslandConfig& island_config);

}  // namespace cpso
