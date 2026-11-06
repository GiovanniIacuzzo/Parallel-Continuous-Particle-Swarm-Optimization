#include "cpso/optimizer.hpp"
#include "cpso/thread_pool.hpp"
#include "optimizer_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cpso {
using namespace detail;
namespace {
struct EvaluationTimings {
    double wall_seconds{};
    double worker_seconds{};
};

void evaluate_serial_batch(
    const Objective& objective,
    const std::vector<std::vector<double>>& positions,
    std::vector<double>& values,
    EvaluationTimings& timings) {
    const auto wall_start = Clock::now();
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto task_start = Clock::now();
        values[i] = checked_evaluate(objective, positions[i]);
        timings.worker_seconds += std::chrono::duration<double>(Clock::now() - task_start).count();
    }
    timings.wall_seconds += std::chrono::duration<double>(Clock::now() - wall_start).count();
}

void evaluate_parallel_batch(
    ThreadPool& pool,
    const Objective& objective,
    const std::vector<std::vector<double>>& positions,
    std::vector<double>& values,
    EvaluationTimings& timings) {
    const auto wall_start = Clock::now();
    std::vector<double> task_seconds(positions.size(), 0.0);
    pool.parallel_for(positions.size(), [&](const std::size_t i) {
        const auto task_start = Clock::now();
        values[i] = checked_evaluate(objective, positions[i]);
        task_seconds[i] = std::chrono::duration<double>(Clock::now() - task_start).count();
    });
    timings.wall_seconds += std::chrono::duration<double>(Clock::now() - wall_start).count();
    for (const double seconds : task_seconds) {
        timings.worker_seconds += seconds;
    }
}

template <typename BatchEvaluator>
[[nodiscard]] Result minimize_synchronous_impl(
    const Config& config,
    const Objective& objective,
    BatchEvaluator&& evaluate_batch,
    const std::size_t worker_count,
    ThreadPool* particle_pool) {
    validate_config(config);
    if (!objective) {
        throw std::invalid_argument("Objective function must be callable");
    }

    const auto start = Clock::now();
    const std::size_t dimension = config.lower_bounds.size();
    const std::size_t population_size = config.population_size;
    std::vector<Particle> particles(population_size);

    Result result;
    result.worker_count = worker_count;
    result.island_count = 1U;
    result.best_history.reserve(config.iterations + 1U);
    result.history_evaluations.reserve(config.iterations + 1U);
    result.best_position.assign(dimension, 0.0);

    std::vector<std::vector<double>> pending_positions(
        population_size, std::vector<double>(dimension));
    std::vector<std::vector<double>> pending_velocities(
        population_size, std::vector<double>(dimension));
    const auto topology_setup_start = Clock::now();
    const ParticleTopology topology(
        config.topology, population_size, config.grid_rows, config.grid_columns,
        config.seed, config.topology_degree, config.topology_rewiring_probability);
    result.topology_setup_seconds =
        std::chrono::duration<double>(Clock::now() - topology_setup_start).count();
    std::vector<double> pending_values(population_size, std::numeric_limits<double>::infinity());

    const auto initialization_start = Clock::now();
    const auto initialize_one = [&](const std::size_t i) {
        initialize_particle(particles[i], config, dimension, static_cast<std::uint64_t>(i + 1U));
        pending_positions[i] = particles[i].position;
    };
    if (particle_pool != nullptr) {
        particle_pool->parallel_for(population_size, initialize_one);
    } else {
        for (std::size_t i = 0; i < population_size; ++i) {
            initialize_one(i);
        }
    }
    result.initialization_seconds =
        std::chrono::duration<double>(Clock::now() - initialization_start).count();

    EvaluationTimings evaluation_timings;
    evaluate_batch(pending_positions, pending_values, evaluation_timings);
    result.function_evaluations += static_cast<std::uint64_t>(population_size);

    // Stable tie-breaking is by particle index.
    for (std::size_t i = 0; i < population_size; ++i) {
        Particle& particle = particles[i];
        particle.current_value = pending_values[i];
        particle.personal_best = particle.position;
        particle.personal_best_value = particle.current_value;
        if (i == 0U || particle.personal_best_value < result.best_value) {
            result.best_value = particle.personal_best_value;
            result.best_position = particle.personal_best;
        }
    }
    result.best_history.push_back(result.best_value);
    result.history_evaluations.push_back(result.function_evaluations);

    std::size_t best_index = global_best_index(particles);
    for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
        std::vector<double> global_best_snapshot;
        std::vector<std::vector<double>> local_snapshot;
        const auto social_start = Clock::now();
        if (config.topology == TopologyKind::GlobalBest) {
            global_best_snapshot = particles[best_index].personal_best;
        } else {
            local_snapshot = local_social_snapshot(particles, topology);
        }
        result.social_selection_seconds +=
            std::chrono::duration<double>(Clock::now() - social_start).count();

        std::vector<double> propagation_task_seconds(population_size, 0.0);
        const auto propagation_start = Clock::now();
        const auto propagate_one = [&](const std::size_t i) {
            const auto task_start = Clock::now();
            const std::vector<double>& attractor = config.topology == TopologyKind::GlobalBest
                ? global_best_snapshot : local_snapshot[i];
            propagate_particle(
                particles[i], attractor, config, iteration,
                pending_positions[i], pending_velocities[i]);
            propagation_task_seconds[i] =
                std::chrono::duration<double>(Clock::now() - task_start).count();
        };
        if (particle_pool != nullptr) {
            particle_pool->parallel_for(population_size, propagate_one);
        } else {
            for (std::size_t i = 0; i < population_size; ++i) {
                propagate_one(i);
            }
        }
        result.propagation_seconds +=
            std::chrono::duration<double>(Clock::now() - propagation_start).count();
        for (const double seconds : propagation_task_seconds) {
            result.propagation_worker_seconds += seconds;
        }

        evaluate_batch(pending_positions, pending_values, evaluation_timings);
        result.function_evaluations += static_cast<std::uint64_t>(population_size);

        for (std::size_t i = 0; i < population_size; ++i) {
            Particle& particle = particles[i];
            particle.position = pending_positions[i];
            particle.velocity = pending_velocities[i];
            particle.current_value = pending_values[i];
            if (particle.current_value < particle.personal_best_value) {
                particle.personal_best = particle.position;
                particle.personal_best_value = particle.current_value;
            }
            if (particle.personal_best_value < result.best_value) {
                result.best_value = particle.personal_best_value;
                result.best_position = particle.personal_best;
            }
        }
        best_index = global_best_index(particles);
        result.iterations_completed = iteration + 1U;
        result.best_history.push_back(result.best_value);
        result.history_evaluations.push_back(result.function_evaluations);
    }

    result.evaluation_wall_seconds = evaluation_timings.wall_seconds;
    result.evaluation_worker_seconds = evaluation_timings.worker_seconds;
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

}  // namespace

void validate_config(const Config& config) {
    if (config.lower_bounds.empty()) {
        throw std::invalid_argument("At least one decision variable is required");
    }
    if (config.lower_bounds.size() != config.upper_bounds.size()) {
        throw std::invalid_argument("Lower and upper bounds must have the same dimension");
    }
    if (config.population_size == 0U) {
        throw std::invalid_argument("population_size must be >= 1");
    }
    if (config.iterations > static_cast<std::size_t>(std::numeric_limits<int>::max() - 1)) {
        throw std::invalid_argument("iterations exceeds the supported range");
    }
    for (std::size_t j = 0; j < config.lower_bounds.size(); ++j) {
        const double lower = config.lower_bounds[j];
        const double upper = config.upper_bounds[j];
        if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)
            || !std::isfinite(upper - lower)) {
            throw std::invalid_argument("Every bound must be finite, have finite width, and satisfy lower < upper");
        }
    }
    if (!std::isfinite(config.mu_min) || !std::isfinite(config.mu_max)
        || config.mu_min < 0.0 || config.mu_max > 1.0 || config.mu_min > config.mu_max) {
        throw std::invalid_argument("Require 0 <= mu_min <= mu_max <= 1");
    }
    if (!std::isfinite(config.cognitive) || !std::isfinite(config.social)
        || config.cognitive < 0.0 || config.social < 0.0
        || !std::isfinite(config.cognitive + config.social)
        || config.cognitive + config.social <= 0.0) {
        throw std::invalid_argument("cognitive and social coefficients must be finite, non-negative, and not both zero");
    }
    if (!std::isfinite(config.dt) || config.dt <= 0.0) {
        throw std::invalid_argument("dt must be finite and > 0");
    }
    if (!std::isfinite(config.initial_velocity_fraction) || config.initial_velocity_fraction < 0.0) {
        throw std::invalid_argument("initial_velocity_fraction must be finite and >= 0");
    }
    for (std::size_t j = 0; j < config.lower_bounds.size(); ++j) {
        if (!std::isfinite(config.initial_velocity_fraction
                            * (config.upper_bounds[j] - config.lower_bounds[j]))) {
            throw std::invalid_argument("initial velocity range overflows for at least one coordinate");
        }
    }
    if (!std::isfinite(config.boundary_restitution)
        || config.boundary_restitution < 0.0 || config.boundary_restitution > 1.0) {
        throw std::invalid_argument("boundary_restitution must lie in [0,1]");
    }
    if (!std::isfinite(config.topology_rewiring_probability)
        || config.topology_rewiring_probability < 0.0
        || config.topology_rewiring_probability > 1.0) {
        throw std::invalid_argument("topology_rewiring_probability must lie in [0,1]");
    }
    switch (config.topology) {
        case TopologyKind::GlobalBest:
        case TopologyKind::Ring:
            if (config.grid_rows != 0U || config.grid_columns != 0U) {
                throw std::invalid_argument("grid dimensions are only valid for grid topology");
            }
            break;
        case TopologyKind::GridVonNeumann:
            if (config.grid_rows < 2U || config.grid_columns < 2U) {
                throw std::invalid_argument("grid topology requires grid_rows >= 2 and grid_columns >= 2");
            }
            if (config.grid_rows > std::numeric_limits<std::size_t>::max() / config.grid_columns
                || config.grid_rows * config.grid_columns != config.population_size) {
                throw std::invalid_argument("grid_rows * grid_columns must equal population_size");
            }
            break;
        case TopologyKind::Wheel:
            if (config.grid_rows != 0U || config.grid_columns != 0U || config.population_size < 2U) {
                throw std::invalid_argument("wheel topology requires N >= 2 and no grid dimensions");
            }
            break;
        case TopologyKind::RandomConnected:
            if (config.grid_rows != 0U || config.grid_columns != 0U
                || config.population_size < 3U || config.topology_degree == 0U
                || config.topology_degree >= config.population_size) {
                throw std::invalid_argument("random topology requires N >= 3 and 1 <= topology_degree < N");
            }
            break;
        case TopologyKind::SmallWorld:
            if (config.grid_rows != 0U || config.grid_columns != 0U
                || config.population_size < 4U || config.topology_degree < 2U
                || config.topology_degree >= config.population_size || config.topology_degree % 2U != 0U) {
                throw std::invalid_argument("small-world topology requires even 2 <= topology_degree < N and N >= 4");
            }
            break;
        case TopologyKind::ScaleFree:
            if (config.grid_rows != 0U || config.grid_columns != 0U
                || config.topology_degree < 2U || config.topology_degree % 2U != 0U
                || config.population_size < config.topology_degree / 2U + 1U) {
                throw std::invalid_argument("scale-free topology requires even topology_degree >= 2 and N >= topology_degree/2 + 1");
            }
            break;
        default:
            throw std::invalid_argument("Unsupported particle topology");
    }
}

Result minimize_serial(const Config& config, const Objective& objective) {
    const auto serial_evaluator = [&objective](
        const std::vector<std::vector<double>>& positions,
        std::vector<double>& values,
        EvaluationTimings& timings) {
        evaluate_serial_batch(objective, positions, values, timings);
    };
    return minimize_synchronous_impl(config, objective, serial_evaluator, 1U, nullptr);
}

Result minimize_master_slave(
    const Config& config,
    const Objective& objective,
    const std::size_t worker_count) {
    validate_config(config);
    if (worker_count == 0U || worker_count > 4096U) {
        throw std::invalid_argument("worker_count must lie in [1,4096]");
    }
    if (!objective) {
        throw std::invalid_argument("Objective function must be callable");
    }
    const auto total_start = Clock::now();
    Result result;
    {
        ThreadPool pool(worker_count);
        const auto parallel_evaluator = [&objective, &pool](
            const std::vector<std::vector<double>>& positions,
            std::vector<double>& values,
            EvaluationTimings& timings) {
            evaluate_parallel_batch(pool, objective, positions, values, timings);
        };
        result = minimize_synchronous_impl(config, objective, parallel_evaluator, worker_count, nullptr);
    }
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

Result minimize_particle_parallel(
    const Config& config,
    const Objective& objective,
    const std::size_t worker_count) {
    validate_config(config);
    if (worker_count == 0U || worker_count > 4096U) {
        throw std::invalid_argument("worker_count must lie in [1,4096]");
    }
    if (!objective) {
        throw std::invalid_argument("Objective function must be callable");
    }
    const auto total_start = Clock::now();
    Result result;
    {
        ThreadPool pool(worker_count);
        const auto parallel_evaluator = [&objective, &pool](
            const std::vector<std::vector<double>>& positions,
            std::vector<double>& values,
            EvaluationTimings& timings) {
            evaluate_parallel_batch(pool, objective, positions, values, timings);
        };
        result = minimize_synchronous_impl(config, objective, parallel_evaluator, worker_count, &pool);
    }
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

}  // namespace cpso
