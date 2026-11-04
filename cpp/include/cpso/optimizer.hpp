#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace cpso {

using Objective = std::function<double(const std::vector<double>&)>;

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
};

struct Result {
    double best_value{};
    std::vector<double> best_position;
    // Includes the initial incumbent, then one value after each completed
    // synchronous propagation/evaluation batch.
    std::vector<double> best_history;
    std::uint64_t function_evaluations{};
    std::size_t iterations_completed{};
    double elapsed_seconds{};
    // Wall-clock time spent in objective-evaluation batches, including
    // dispatch/wait overhead. Timed once per batch to keep per-call overhead out.
    double evaluation_wall_seconds{};
    double propagation_seconds{};
    std::size_t worker_count{1};
};

void validate_config(const Config& config);
[[nodiscard]] Result minimize_serial(const Config& config, const Objective& objective);

// Synchronous master--slave variant. Particle propagation, random-number
// generation, best-state commits, and social snapshots are identical to the
// serial reference; only objective evaluations are dispatched concurrently.
// The objective must be reentrant/thread-safe. Results are collected and
// committed in particle-index order to preserve deterministic tie-breaking.
[[nodiscard]] Result minimize_master_slave(
    const Config& config,
    const Objective& objective,
    std::size_t worker_count);

}  // namespace cpso
