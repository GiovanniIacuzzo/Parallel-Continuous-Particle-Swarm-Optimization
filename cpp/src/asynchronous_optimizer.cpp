#include "cpso/optimizer.hpp"
#include "cpso/thread_pool.hpp"
#include "optimizer_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <queue>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cpso {
using namespace detail;
namespace {
struct EvaluationTimings {
    double wall_seconds{};
    double worker_seconds{};
};

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
}  // namespace

Result minimize_asynchronous(
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
    if (config.iterations > 0U
        && config.population_size > std::numeric_limits<std::size_t>::max() / config.iterations) {
        throw std::invalid_argument("Asynchronous update count overflows size_t");
    }

    const auto total_start = Clock::now();
    const std::size_t dimension = config.lower_bounds.size();
    const std::size_t population_size = config.population_size;
    Result result;
    result.worker_count = worker_count;
    result.island_count = 1U;
    result.best_position.assign(dimension, 0.0);
    const std::size_t update_count = population_size * config.iterations;
    if (!config.summary_only) {
        result.best_history.reserve(1U + update_count);
        result.history_evaluations.reserve(1U + update_count);
    }
    const bool timed = config.wall_time_limit_seconds > 0.0;
    const auto deadline = total_start + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(config.wall_time_limit_seconds));


    const auto topology_start = Clock::now();
    const ParticleTopology topology(
        config.topology, population_size, config.grid_rows, config.grid_columns,
        config.seed, config.topology_degree, config.topology_rewiring_probability);
    result.topology_setup_seconds =
        std::chrono::duration<double>(Clock::now() - topology_start).count();

    std::vector<Particle> particles(population_size);
    std::vector<std::vector<double>> initial_positions(
        population_size, std::vector<double>(dimension));
    std::vector<double> initial_values(population_size, std::numeric_limits<double>::infinity());
    EvaluationTimings initial_evaluations;
    {
        ThreadPool initialization_pool(worker_count);
        const auto initialization_start = Clock::now();
        initialization_pool.parallel_for(population_size, [&](const std::size_t i) {
            initialize_particle(particles[i], config, dimension, static_cast<std::uint64_t>(i + 1U));
            initial_positions[i] = particles[i].position;
        });
        result.initialization_seconds =
            std::chrono::duration<double>(Clock::now() - initialization_start).count();
        evaluate_parallel_batch(
            initialization_pool, objective, initial_positions, initial_values, initial_evaluations);
    }
    result.evaluation_wall_seconds += initial_evaluations.wall_seconds;
    result.evaluation_worker_seconds += initial_evaluations.worker_seconds;
    result.function_evaluations = static_cast<std::uint64_t>(population_size);

    std::size_t global_best_particle = 0U;
    for (std::size_t i = 0; i < population_size; ++i) {
        particles[i].current_value = initial_values[i];
        particles[i].personal_best = particles[i].position;
        particles[i].personal_best_value = initial_values[i];
        if (i == 0U || initial_values[i] < result.best_value) {
            result.best_value = initial_values[i];
            result.best_position = particles[i].personal_best;
            global_best_particle = i;
        }
    }
    if (!config.summary_only) {
        result.best_history.push_back(result.best_value);
        result.history_evaluations.push_back(result.function_evaluations);
    }

    struct Event {
        std::size_t particle_index{};
        std::size_t local_iteration{};
    };
    std::mutex state_mutex;
    std::vector<double> propagation_task_seconds(population_size, 0.0);
    std::vector<double> evaluation_task_seconds(population_size, 0.0);
    std::vector<double> social_task_seconds(population_size, 0.0);
    std::uint64_t committed_updates = 0U;
    // All control state is protected by state_mutex, never by a per-worker lock.
    double active_dt = config.dt;
    const std::uint64_t stagnation_window = std::max<std::uint64_t>(
        500U, 12U * static_cast<std::uint64_t>(population_size));
    std::uint64_t last_global_improvement = 0U;
    std::uint64_t last_recovery = 0U;
    std::vector<bool> recovery_pending(population_size, false);
    std::atomic<bool> expired{false};


    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::queue<Event> ready_events;
    for (std::size_t i = 0; i < population_size; ++i) {
        ready_events.push(Event{i, 0U});
    }
    std::size_t remaining_updates = update_count;
    bool stopping = (update_count == 0U);
    std::exception_ptr first_exception;

    const auto execute_event = [&](const Event event) {
        const std::size_t particle_index = event.particle_index;
        Particle& particle = particles[particle_index];
        std::vector<double> own_personal_best(dimension);
        std::vector<double> social_attractor(dimension);
        double step_dt = config.dt;
        bool recover_this_particle = false;


        const auto social_start = Clock::now();
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            own_personal_best = particle.personal_best;
            step_dt = active_dt;
            if (config.adaptive_recovery && recovery_pending[particle_index]) {
                recovery_pending[particle_index] = false;
                recover_this_particle = particle_index != global_best_particle;
            }
            if (config.topology == TopologyKind::GlobalBest) {
                social_attractor = result.best_position;
            } else {
                const auto& neighbors = topology.neighbors(particle_index);
                std::size_t leader = neighbors.front();
                for (const std::size_t candidate : neighbors) {
                    if (particles[candidate].personal_best_value
                            < particles[leader].personal_best_value
                        || (particles[candidate].personal_best_value
                                == particles[leader].personal_best_value
                            && candidate < leader)) {
                        leader = candidate;
                    }
                }
                social_attractor = particles[leader].personal_best;
            }
        }
        social_task_seconds[particle_index] +=
            std::chrono::duration<double>(Clock::now() - social_start).count();

        std::vector<double> candidate_position(dimension);
        std::vector<double> candidate_velocity(dimension);
        const auto propagation_start = Clock::now();
        if (recover_this_particle) {
            // Restart counts as exactly one regular FE: candidate evaluated below.
            // Half of restarts sample globally; half sample around elite position.
            const bool global_restart = (particle_index % 2U == 0U);
            for (std::size_t j = 0; j < dimension; ++j) {
                const double lo = config.lower_bounds[j];
                const double hi = config.upper_bounds[j];
                const double width = hi - lo;
                const double centre = global_restart ? (lo + 0.5 * width)
                                                    : social_attractor[j];
                const double radius = global_restart ? 0.5 * width : 0.2 * width;
                const double proposed = centre + (2.0 * uniform01(particle.random_engine)-1.0)*radius;
                candidate_position[j] = std::clamp(proposed, lo, hi);
                candidate_velocity[j] = 0.0;
            }
        } else {
        const double r1 = uniform01(particle.random_engine);
        const double r2 = uniform01(particle.random_engine);
        const double cognitive_weight = config.cognitive * r1;
        const double social_weight = config.social * r2;
        const double omega2 = cognitive_weight + social_weight;
        const double mu = inertia_at(config, event.local_iteration);
        for (std::size_t j = 0; j < dimension; ++j) {
            double equilibrium = 0.0;
            if (omega2 > 0.0) {
                equilibrium = (cognitive_weight * own_personal_best[j]
                               + social_weight * social_attractor[j]) / omega2;
            }
            OscillatorState next = propagate_frozen_oscillator(
                particle.position[j], particle.velocity[j], equilibrium,
                omega2, mu, step_dt);
            apply_clamped_restitution(
                next.position, next.velocity,
                config.lower_bounds[j], config.upper_bounds[j],
                config.boundary_restitution);
            candidate_position[j] = next.position;
            candidate_velocity[j] = next.velocity;
        }
        }  // end regular propagation / recovery branch
        propagation_task_seconds[particle_index] +=
            std::chrono::duration<double>(Clock::now() - propagation_start).count();

        const auto evaluation_start = Clock::now();
        const double candidate_value = checked_evaluate(objective, candidate_position);
        evaluation_task_seconds[particle_index] +=
            std::chrono::duration<double>(Clock::now() - evaluation_start).count();
        // Only the worker that owns this event writes the particle's current state.
        particle.position = std::move(candidate_position);
        particle.velocity = std::move(candidate_velocity);

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            // Do not count a candidate completed after the hard wall-clock deadline.
            if (timed && Clock::now() >= deadline) {
                expired = true;
                return;
            }
            const double best_before = result.best_value;
            if (recover_this_particle) {
                // A restart forgets the particle's old local attractor but never
                // deletes the global best archive (global-best particle is exempt).
                particle.personal_best = particle.position;
                particle.personal_best_value = candidate_value;
            }
            if (candidate_value < particle.personal_best_value) {
                particle.personal_best = particle.position;
                particle.personal_best_value = candidate_value;
            }
            if (particle.personal_best_value < result.best_value
                || (particle.personal_best_value == result.best_value
                    && particle_index < global_best_particle)) {
                result.best_value = particle.personal_best_value;
                result.best_position = particle.personal_best;
                global_best_particle = particle_index;
            }
            ++committed_updates;
            if (config.adaptive_recovery) {
                if (result.best_value < best_before) {
                    last_global_improvement = committed_updates;
                    active_dt = std::max(config.dt, active_dt * 0.85);
                }
                if (committed_updates - last_global_improvement >= stagnation_window
                    && committed_updates - last_recovery >= stagnation_window) {
                    last_recovery = committed_updates;
                    active_dt = std::min(3.0 * config.dt, active_dt * 1.25);
                    // Schedule the worst fifth of the swarm for recovery.
                    std::vector<std::size_t> indices(population_size);
                    std::iota(indices.begin(), indices.end(), 0U);
                    std::sort(indices.begin(), indices.end(), [&](std::size_t x, std::size_t y) {
                        if (particles[x].personal_best_value != particles[y].personal_best_value)
                            return particles[x].personal_best_value > particles[y].personal_best_value;
                        return x < y;
                    });
                    const std::size_t count = std::max<std::size_t>(1, population_size / 5);
                    std::size_t scheduled = 0;
                    for (const auto i : indices) {
                        if (i != global_best_particle && !recovery_pending[i]) {
                            recovery_pending[i] = true;
                            if (++scheduled >= count) break;
                        }
                    }
                }
            }
            if (!config.summary_only) {
                result.best_history.push_back(result.best_value);
                result.history_evaluations.push_back(
                    static_cast<std::uint64_t>(population_size) + committed_updates);
            }
        }
    };

    const auto async_start = Clock::now();
    if (update_count > 0U) {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        try {
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                workers.emplace_back([&] {
                    while (true) {
                        Event event;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            queue_condition.wait(lock, [&] {
                                return stopping || !ready_events.empty();
                            });
                            if (stopping || expired || (timed && Clock::now() >= deadline)) {
                                stopping = true;
                                queue_condition.notify_all();
                                return;
                            }
                            event = ready_events.front();
                            ready_events.pop();
                        }

                        try {
                            execute_event(event);
                        } catch (...) {
                            {
                                std::lock_guard<std::mutex> lock(queue_mutex);
                                if (!first_exception) {
                                    first_exception = std::current_exception();
                                }
                                stopping = true;
                            }
                            queue_condition.notify_all();
                            return;
                        }

                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            if (stopping) {
                                return;
                            }
                            if (expired || (timed && Clock::now() >= deadline)) {
                                stopping = true;
                                queue_condition.notify_all();
                                return;
                            }
                            --remaining_updates;
                            if (remaining_updates == 0U) {
                                stopping = true;
                            } else if (event.local_iteration + 1U < config.iterations) {
                                // Requeue only after this particle's update has
                                // completed: no two workers can mutate one particle.
                                ready_events.push(Event{
                                    event.particle_index, event.local_iteration + 1U});
                            }
                        }
                        queue_condition.notify_all();
                    }
                });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stopping = true;
            }
            queue_condition.notify_all();
            for (std::thread& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
        queue_condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    result.async_update_wall_seconds =
        std::chrono::duration<double>(Clock::now() - async_start).count();
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    if (!timed && committed_updates != static_cast<std::uint64_t>(update_count)) {
        throw std::runtime_error("Asynchronous scheduler stopped before all particle updates completed");
    }
    result.function_evaluations += committed_updates;
    result.iterations_completed = timed ? static_cast<std::size_t>(
        committed_updates / population_size) : config.iterations;
    if (config.summary_only) {
        result.best_history.push_back(result.best_value);
        result.history_evaluations.push_back(result.function_evaluations);
    }
    for (std::size_t i = 0; i < population_size; ++i) {
        result.propagation_worker_seconds += propagation_task_seconds[i];
        result.evaluation_worker_seconds += evaluation_task_seconds[i];
        result.social_selection_seconds += social_task_seconds[i];
    }
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

}  // namespace cpso
