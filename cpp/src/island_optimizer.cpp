#include "cpso/optimizer.hpp"
#include "cpso/thread_pool.hpp"
#include "optimizer_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cpso {
using namespace detail;

namespace {
struct IslandState {
    IslandState(
        const std::size_t local_population,
        const Config& config,
        const std::size_t dimension)
        : topology(config.topology, local_population),
          particles(local_population),
          pending_positions(local_population, std::vector<double>(dimension)),
          pending_velocities(local_population, std::vector<double>(dimension)),
          pending_values(local_population, std::numeric_limits<double>::infinity()) {}

    ParticleTopology topology;
    std::vector<Particle> particles;
    std::vector<std::vector<double>> pending_positions;
    std::vector<std::vector<double>> pending_velocities;
    std::vector<double> pending_values;
    std::vector<double> global_social;
    std::vector<std::vector<double>> local_social;
};

[[nodiscard]] std::size_t island_best_index(const IslandState& island) {
    return global_best_index(island.particles);
}

void build_island_social_snapshot(IslandState& island, const TopologyKind kind) {
    const std::size_t best = island_best_index(island);
    if (kind == TopologyKind::GlobalBest) {
        island.global_social = island.particles[best].personal_best;
        island.local_social.clear();
        return;
    }
    island.global_social.clear();
    island.local_social = local_social_snapshot(island.particles, island.topology);
}

struct Emigrant {
    std::vector<double> position;
    double value{};
};

[[nodiscard]] bool better_migrant(
    const Emigrant& candidate,
    const Emigrant& incumbent,
    const std::size_t candidate_source,
    const std::size_t incumbent_source) {
    return candidate.value < incumbent.value
        || (candidate.value == incumbent.value && candidate_source < incumbent_source);
}
}  // namespace

Result minimize_islands(
    const Config& config,
    const Objective& objective,
    const IslandConfig& island_config) {
    validate_config(config);
    if (!objective) {
        throw std::invalid_argument("Objective function must be callable");
    }
    if (island_config.island_count < 2U || island_config.island_count > config.population_size) {
        throw std::invalid_argument("island_count must lie in [2,population_size]");
    }
    if (config.population_size % island_config.island_count != 0U) {
        throw std::invalid_argument("population_size must be divisible by island_count");
    }
    if (config.population_size / island_config.island_count < 2U) {
        throw std::invalid_argument("Each island must contain at least two particles");
    }
    if (island_config.worker_count == 0U || island_config.worker_count > 4096U) {
        throw std::invalid_argument("worker_count must lie in [1,4096]");
    }
    if (island_config.worker_count > island_config.island_count) {
        throw std::invalid_argument("worker_count cannot exceed island_count in the island backend");
    }
    if (config.topology == TopologyKind::GridVonNeumann) {
        throw std::invalid_argument(
            "Grid particle topology is not supported inside islands; use global or ring");
    }
    if (island_config.migration_topology != MigrationTopology::Ring
        && island_config.migration_topology != MigrationTopology::FullyConnected) {
        throw std::invalid_argument("Unsupported island migration topology");
    }

    const auto total_start = Clock::now();
    const std::size_t dimension = config.lower_bounds.size();
    const std::size_t local_population = config.population_size / island_config.island_count;
    ThreadPool pool(island_config.worker_count);
    const auto topology_start = Clock::now();
    std::vector<IslandState> islands;
    islands.reserve(island_config.island_count);
    for (std::size_t island = 0; island < island_config.island_count; ++island) {
        islands.emplace_back(local_population, config, dimension);
    }
    const double island_topology_setup_seconds =
        std::chrono::duration<double>(Clock::now() - topology_start).count();

    Result result;
    result.worker_count = island_config.worker_count;
    result.island_count = island_config.island_count;
    result.best_position.assign(dimension, 0.0);
    result.best_history.reserve(config.iterations + 1U);
    result.history_evaluations.reserve(config.iterations + 1U);

    result.topology_setup_seconds = island_topology_setup_seconds;

    const auto initialization_start = Clock::now();
    pool.parallel_for(island_config.island_count, [&](const std::size_t island_index) {
        IslandState& island = islands[island_index];
        const std::size_t first_global_index = island_index * local_population;
        for (std::size_t local_index = 0; local_index < local_population; ++local_index) {
            initialize_particle(
                island.particles[local_index], config, dimension,
                static_cast<std::uint64_t>(first_global_index + local_index + 1U));
            island.pending_positions[local_index] = island.particles[local_index].position;
        }
    });
    result.initialization_seconds =
        std::chrono::duration<double>(Clock::now() - initialization_start).count();

    std::vector<double> initial_evaluation_task_seconds(island_config.island_count, 0.0);
    const auto initial_eval_start = Clock::now();
    pool.parallel_for(island_config.island_count, [&](const std::size_t island_index) {
        IslandState& island = islands[island_index];
        const auto task_start = Clock::now();
        for (std::size_t i = 0; i < local_population; ++i) {
            island.pending_values[i] = checked_evaluate(objective, island.pending_positions[i]);
        }
        initial_evaluation_task_seconds[island_index] =
            std::chrono::duration<double>(Clock::now() - task_start).count();
    });
    result.evaluation_wall_seconds +=
        std::chrono::duration<double>(Clock::now() - initial_eval_start).count();
    for (const double seconds : initial_evaluation_task_seconds) {
        result.evaluation_worker_seconds += seconds;
    }

    for (std::size_t island_index = 0; island_index < islands.size(); ++island_index) {
        IslandState& island = islands[island_index];
        for (std::size_t i = 0; i < local_population; ++i) {
            Particle& particle = island.particles[i];
            particle.current_value = island.pending_values[i];
            particle.personal_best = particle.position;
            particle.personal_best_value = particle.current_value;
            if (result.function_evaluations == 0U || particle.personal_best_value < result.best_value) {
                result.best_value = particle.personal_best_value;
                result.best_position = particle.personal_best;
            }
            ++result.function_evaluations;
        }
    }
    result.best_history.push_back(result.best_value);
    result.history_evaluations.push_back(result.function_evaluations);

    std::vector<double> island_phase_seconds(island_config.island_count, 0.0);
    for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
        const auto social_start = Clock::now();
        for (IslandState& island : islands) {
            build_island_social_snapshot(island, config.topology);
        }
        result.social_selection_seconds +=
            std::chrono::duration<double>(Clock::now() - social_start).count();

        std::fill(island_phase_seconds.begin(), island_phase_seconds.end(), 0.0);
        const auto propagation_start = Clock::now();
        pool.parallel_for(island_config.island_count, [&](const std::size_t island_index) {
            IslandState& island = islands[island_index];
            const auto task_start = Clock::now();
            for (std::size_t i = 0; i < local_population; ++i) {
                const std::vector<double>& attractor = config.topology == TopologyKind::GlobalBest
                    ? island.global_social : island.local_social[i];
                propagate_particle(
                    island.particles[i], attractor, config, iteration,
                    island.pending_positions[i], island.pending_velocities[i]);
            }
            island_phase_seconds[island_index] =
                std::chrono::duration<double>(Clock::now() - task_start).count();
        });
        result.propagation_seconds +=
            std::chrono::duration<double>(Clock::now() - propagation_start).count();
        for (const double seconds : island_phase_seconds) {
            result.propagation_worker_seconds += seconds;
        }

        std::fill(island_phase_seconds.begin(), island_phase_seconds.end(), 0.0);
        const auto evaluation_start = Clock::now();
        pool.parallel_for(island_config.island_count, [&](const std::size_t island_index) {
            IslandState& island = islands[island_index];
            const auto task_start = Clock::now();
            for (std::size_t i = 0; i < local_population; ++i) {
                island.pending_values[i] = checked_evaluate(objective, island.pending_positions[i]);
            }
            island_phase_seconds[island_index] =
                std::chrono::duration<double>(Clock::now() - task_start).count();
        });
        result.evaluation_wall_seconds +=
            std::chrono::duration<double>(Clock::now() - evaluation_start).count();
        for (const double seconds : island_phase_seconds) {
            result.evaluation_worker_seconds += seconds;
        }

        for (IslandState& island : islands) {
            for (std::size_t i = 0; i < local_population; ++i) {
                Particle& particle = island.particles[i];
                particle.position = island.pending_positions[i];
                particle.velocity = island.pending_velocities[i];
                particle.current_value = island.pending_values[i];
                if (particle.current_value < particle.personal_best_value) {
                    particle.personal_best = particle.position;
                    particle.personal_best_value = particle.current_value;
                }
                if (particle.personal_best_value < result.best_value) {
                    result.best_value = particle.personal_best_value;
                    result.best_position = particle.personal_best;
                }
            }
        }
        result.function_evaluations += static_cast<std::uint64_t>(config.population_size);
        result.iterations_completed = iteration + 1U;
        result.best_history.push_back(result.best_value);
        result.history_evaluations.push_back(result.function_evaluations);

        const bool migration_due = island_config.migration_interval > 0U
            && (iteration + 1U) % island_config.migration_interval == 0U
            && (iteration + 1U) < config.iterations;
        if (migration_due) {
            const auto migration_start = Clock::now();
            // Freeze every emigrant before applying any replacement. This makes
            // migration a simultaneous epoch-level exchange rather than an
            // order-dependent cascade through islands updated earlier in the loop.
            std::vector<Emigrant> emigrants;
            emigrants.reserve(island_config.island_count);
            for (const IslandState& island : islands) {
                const Particle& best = island.particles[island_best_index(island)];
                emigrants.push_back(Emigrant{best.personal_best, best.personal_best_value});
            }

            for (std::size_t destination = 0; destination < islands.size(); ++destination) {
                std::size_t source = destination;
                if (island_config.migration_topology == MigrationTopology::Ring) {
                    source = (destination + islands.size() - 1U) % islands.size();
                } else {
                    bool has_source = false;
                    for (std::size_t candidate = 0; candidate < islands.size(); ++candidate) {
                        if (candidate == destination) {
                            continue;
                        }
                        if (!has_source
                            || better_migrant(
                                emigrants[candidate], emigrants[source], candidate, source)) {
                            source = candidate;
                            has_source = true;
                        }
                    }
                }

                IslandState& destination_island = islands[destination];
                const Emigrant& emigrant = emigrants[source];
                std::size_t worst = 0U;
                for (std::size_t i = 1U; i < local_population; ++i) {
                    if (destination_island.particles[i].personal_best_value
                        > destination_island.particles[worst].personal_best_value) {
                        worst = i;
                    }
                }
                Particle& recipient = destination_island.particles[worst];
                if (emigrant.value < recipient.personal_best_value) {
                    recipient.position = emigrant.position;
                    recipient.velocity.assign(dimension, 0.0);
                    recipient.personal_best = emigrant.position;
                    recipient.current_value = emigrant.value;
                    recipient.personal_best_value = emigrant.value;
                    ++result.migrations_performed;
                }
            }
            result.migration_seconds +=
                std::chrono::duration<double>(Clock::now() - migration_start).count();
        }
    }

    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

}  // namespace cpso
