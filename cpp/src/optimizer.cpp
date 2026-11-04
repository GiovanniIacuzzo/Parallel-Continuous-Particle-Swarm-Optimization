#include "cpso/optimizer.hpp"

#include "cpso/analytic_propagator.hpp"
#include "cpso/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cpso {
namespace {

using Clock = std::chrono::steady_clock;

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> personal_best;
    double current_value{std::numeric_limits<double>::infinity()};
    double personal_best_value{std::numeric_limits<double>::infinity()};
    std::mt19937_64 random_engine;
};

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] double uniform01(std::mt19937_64& engine) {
    return std::generate_canonical<double, 53>(engine);
}

[[nodiscard]] double inertia_at(const Config& config, const std::size_t iteration) {
    if (config.iterations <= 1U) {
        return config.mu_max;
    }
    const double fraction = static_cast<double>(iteration)
        / static_cast<double>(config.iterations - 1U);
    return config.mu_max - (config.mu_max - config.mu_min) * fraction;
}

[[nodiscard]] double checked_evaluate(
    const Objective& objective,
    const std::vector<double>& position) {
    const double value = objective(position);
    if (!std::isfinite(value)) {
        throw std::runtime_error("Objective function returned a non-finite value");
    }
    return value;
}

struct EvaluationTimings {
    double wall_seconds{};
};

void evaluate_serial_batch(
    const Objective& objective,
    const std::vector<std::vector<double>>& positions,
    std::vector<double>& values,
    EvaluationTimings& timings) {
    const auto wall_start = Clock::now();
    for (std::size_t i = 0; i < positions.size(); ++i) {
        values[i] = checked_evaluate(objective, positions[i]);
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
    pool.parallel_for(positions.size(), [&](const std::size_t i) {
        values[i] = checked_evaluate(objective, positions[i]);
    });
    timings.wall_seconds += std::chrono::duration<double>(Clock::now() - wall_start).count();
}

template <typename BatchEvaluator>
[[nodiscard]] Result minimize_impl(
    const Config& config,
    const Objective& objective,
    BatchEvaluator&& evaluate_batch,
    const std::size_t worker_count) {
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
    result.best_history.reserve(config.iterations + 1U);
    result.best_position.assign(dimension, 0.0);

    // Generate the complete initial population before evaluating it. Each
    // particle owns an independent stream, so this preserves the serial
    // reference sequence while permitting an evaluation batch to run in parallel.
    std::vector<std::vector<double>> pending_positions(
        population_size, std::vector<double>(dimension));
    std::vector<std::vector<double>> pending_velocities(
        population_size, std::vector<double>(dimension));
    std::vector<double> pending_values(population_size, std::numeric_limits<double>::infinity());

    for (std::size_t i = 0; i < population_size; ++i) {
        Particle& particle = particles[i];
        particle.random_engine.seed(splitmix64(config.seed ^ static_cast<std::uint64_t>(i + 1U)));
        particle.position.resize(dimension);
        particle.velocity.resize(dimension);
        particle.personal_best.resize(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            const double lower = config.lower_bounds[j];
            const double upper = config.upper_bounds[j];
            const double width = upper - lower;
            particle.position[j] = lower + uniform01(particle.random_engine) * width;
            particle.velocity[j] = (2.0 * uniform01(particle.random_engine) - 1.0)
                                   * config.initial_velocity_fraction * width;
        }
        pending_positions[i] = particle.position;
    }

    EvaluationTimings evaluation_timings;
    evaluate_batch(pending_positions, pending_values, evaluation_timings);
    result.function_evaluations += static_cast<std::uint64_t>(population_size);

    // Stable tie-breaking is by particle index in both execution modes.
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

    // Synchronous generation semantics: freeze one gbest snapshot, propagate
    // every particle serially, evaluate the full batch, and only then commit.
    for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
        const std::vector<double> global_best_snapshot = result.best_position;
        const double mu = inertia_at(config, iteration);
        const auto propagation_start = Clock::now();

        for (std::size_t i = 0; i < population_size; ++i) {
            Particle& particle = particles[i];
            const double r1 = uniform01(particle.random_engine);
            const double r2 = uniform01(particle.random_engine);
            const double cognitive_weight = config.cognitive * r1;
            const double social_weight = config.social * r2;
            const double omega2 = cognitive_weight + social_weight;

            for (std::size_t j = 0; j < dimension; ++j) {
                double equilibrium = 0.0;
                if (omega2 > 0.0) {
                    equilibrium = (cognitive_weight * particle.personal_best[j]
                                   + social_weight * global_best_snapshot[j]) / omega2;
                }

                OscillatorState next = propagate_frozen_oscillator(
                    particle.position[j], particle.velocity[j], equilibrium,
                    omega2, mu, config.dt);
                apply_clamped_restitution(
                    next.position, next.velocity,
                    config.lower_bounds[j], config.upper_bounds[j],
                    config.boundary_restitution);
                pending_positions[i][j] = next.position;
                pending_velocities[i][j] = next.velocity;
            }
        }
        result.propagation_seconds +=
            std::chrono::duration<double>(Clock::now() - propagation_start).count();

        evaluate_batch(pending_positions, pending_values, evaluation_timings);
        result.function_evaluations += static_cast<std::uint64_t>(population_size);

        // Commit in particle order: deterministic tie-breaking is independent
        // of the order in which objective tasks completed on worker threads.
        for (std::size_t i = 0; i < population_size; ++i) {
            Particle& particle = particles[i];
            particle.position = pending_positions[i];
            particle.velocity = pending_velocities[i];
            particle.current_value = pending_values[i];
            if (particle.current_value < particle.personal_best_value) {
                particle.personal_best_value = particle.current_value;
                particle.personal_best = particle.position;
            }
            if (particle.personal_best_value < result.best_value) {
                result.best_value = particle.personal_best_value;
                result.best_position = particle.personal_best;
            }
        }
        result.iterations_completed = iteration + 1U;
        result.best_history.push_back(result.best_value);
    }

    result.evaluation_wall_seconds = evaluation_timings.wall_seconds;
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
}

Result minimize_serial(const Config& config, const Objective& objective) {
    const auto serial_evaluator = [&objective](
        const std::vector<std::vector<double>>& positions,
        std::vector<double>& values,
        EvaluationTimings& timings) {
        evaluate_serial_batch(objective, positions, values, timings);
    };
    return minimize_impl(config, objective, serial_evaluator, 1U);
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
        result = minimize_impl(config, objective, parallel_evaluator, worker_count);
    }  // Include pool creation and shutdown in total wall-clock time.
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

}  // namespace cpso
