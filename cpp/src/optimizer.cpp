#include "cpso/optimizer.hpp"
#include "cpso/thread_pool.hpp"
#include "optimizer_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cpso {
using namespace detail;
namespace {
struct EvaluationTimings {
    double wall_seconds{};
    double worker_seconds{};
};

enum class TrialKind { Swarm, Local, Quadratic, RestartGlobal, RestartNearLeader };

struct RefinementTrial {
    TrialKind kind{TrialKind::Swarm};
    std::size_t axis{};
    double sign{};
};

struct Refinement {
    std::vector<RefinementTrial> trials;
    std::vector<double> radius;
    std::vector<std::size_t> available;
    std::vector<std::size_t> personal_progress;
    std::vector<double> quadratic_point;
    bool quadratic_pending{};
    std::size_t next_axis{};
    std::size_t last_progress{};
    std::size_t last_restart{};

    Refinement(std::size_t batch, std::size_t dimension, std::size_t population)
        : trials(batch), radius(dimension, 0.02), personal_progress(population) {
        available.reserve(batch);
    }

    void plan(const SynchronousStudy& study, const std::vector<Particle>& particles,
              std::size_t best, std::size_t first, std::size_t batch, std::size_t sweep_batches) {
        std::fill(trials.begin(), trials.end(), RefinementTrial{});
        const auto stalled = batch - last_progress;
        const bool local = study.local_search && batch >= 10U * sweep_batches
            && (stalled >= 6U * sweep_batches || batch % (2U * sweep_batches) == 0U);
        const bool restart = study.partial_restart && batch >= 16U * sweep_batches
            && batch - last_restart >= 8U * sweep_batches;
        if (!local && !restart && !quadratic_pending) return;

        available.clear();
        for (std::size_t slot = 0; slot < trials.size(); ++slot) {
            if (first + slot != best) available.push_back(slot);
        }
        std::sort(available.begin(), available.end(), [&](auto a, auto b) {
            const auto va = particles[first + a].personal_best_value;
            const auto vb = particles[first + b].personal_best_value;
            return va == vb ? a < b : va > vb;
        });

        std::size_t used = 0;
        if (local) {
            const auto pairs = (available.size() - (quadratic_pending && !available.empty())) / 2U;
            const auto axes = std::min({radius.size(), std::size_t{4}, pairs});
            for (std::size_t k = 0; k < axes; ++k) {
                const auto axis = next_axis;
                next_axis = (next_axis + 1U) % radius.size();
                trials[available[used++]] = {TrialKind::Local, axis, -1.0};
                trials[available[used++]] = {TrialKind::Local, axis, 1.0};
            }
        }
        if (quadratic_pending && used < available.size()) {
            trials[available[used++]] = {TrialKind::Quadratic};
        }
        quadratic_pending = false;
        if (restart) {
            const auto limit = std::max(std::size_t{1}, trials.size() / 10U);
            const auto weak_end = (available.size() + 1U) / 2U;
            std::size_t count = 0;
            for (; used < weak_end && count < limit; ++used) {
                const auto slot = available[used];
                if (batch - personal_progress[first + slot] < 16U * sweep_batches) continue;
                trials[slot] = {
                    count % 2U == 0U ? TrialKind::RestartGlobal : TrialKind::RestartNearLeader
                };
                ++count;
            }
            if (count != 0U) last_restart = batch;
        }
    }

    void finish_local(const Config& config, const std::vector<std::vector<double>>& positions,
                      const std::vector<double>& values, const std::vector<double>& leader,
                      double incumbent) {
        if (available.empty() || trials[available.front()].kind != TrialKind::Local) return;
        // Only schedule a prediction that could improve on the best evaluated point.
        double best_prediction = config.study->quadratic_local
            ? std::min(incumbent, *std::min_element(values.begin(), values.end())) - incumbent
            : 0.0;
        for (std::size_t k = 0; k + 1U < available.size(); k += 2U) {
            const auto first = available[k];
            const auto second = available[k + 1U];
            if (trials[first].kind != TrialKind::Local) break;
            const auto axis = trials[first].axis;
            auto& step = radius[axis];
            if (config.study->quadratic_local) {
                // Fit changes from the old leader in normalized coordinates -1, 0, +1.
                const auto width = config.upper_bounds[axis] - config.lower_bounds[axis];
                const auto distance = width * step;
                if (positions[first][axis] == leader[axis] - distance
                    && positions[second][axis] == leader[axis] + distance
                    && positions[first][axis] < leader[axis] && leader[axis] < positions[second][axis]) {
                    const auto minus = values[first] - incumbent;
                    const auto plus = values[second] - incumbent;
                    const auto curvature = 0.5 * minus + 0.5 * plus;
                    const auto slope = 0.5 * plus - 0.5 * minus;
                    if (std::isfinite(curvature) && curvature > 0.0) {
                        const auto fraction = -0.5 * slope / curvature;
                        const auto predicted_change = 0.5 * slope * fraction;
                        if (std::abs(fraction) >= 0.05 && std::abs(fraction) <= 0.90
                            && std::isfinite(predicted_change) && predicted_change < best_prediction) {
                            best_prediction = predicted_change;
                            quadratic_point = leader;
                            quadratic_point[axis] += fraction * distance;
                            quadratic_pending = true;
                        }
                    }
                }
            }
            const bool improved = std::min(values[first], values[second]) < incumbent;
            step = std::clamp(step * (improved ? 1.25 : 0.5), 1e-10, 0.10);
        }
    }
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
    ThreadPool* particle_pool, std::uint64_t graph_seed) {
    validate_config(config);
    if (!objective) {
        throw std::invalid_argument("Objective function must be callable");
    }

    auto* study = config.study;
    const auto start = study == nullptr ? Clock::now() : study->start;
    const auto elapsed = [&] {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    const auto expired = [&](const double seconds) {
        return study != nullptr && study->seconds > 0.0 && seconds >= study->seconds;
    };
    const std::size_t dimension = config.lower_bounds.size();
    const std::size_t population_size = config.population_size;
    const std::size_t update_batch_size = study != nullptr && study->update_batch_size != 0U
        ? study->update_batch_size : population_size;
    if (population_size % update_batch_size != 0U
        || (update_batch_size != population_size && (study->adaptive_step || study->recovery))) {
        throw std::invalid_argument("Invalid synchronous update batch");
    }
    const std::size_t batches_per_sweep = population_size / update_batch_size;
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
        config.graph_seed.value_or(graph_seed), config.topology_degree, config.topology_rewiring_probability);
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
    result.best_value = std::numeric_limits<double>::infinity();
    const auto accept_batch = [&] {
        if (study != nullptr && expired(elapsed())) return false;
        evaluate_batch(pending_positions, pending_values, evaluation_timings);
        if (study != nullptr) {
            const double seconds = elapsed();
            if (expired(seconds)) {
                study->discarded_evaluations = pending_positions.size();
                return false;
            }
            study->batch_seconds.push_back(seconds);
        }
        result.function_evaluations += pending_positions.size();
        return true;
    };
    if (!accept_batch()) {
        result.elapsed_seconds = elapsed();
        return result;
    }

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
    std::optional<Refinement> refinement;
    if (study != nullptr && (study->local_search || study->partial_restart)) {
        refinement.emplace(update_batch_size, dimension, population_size);
    }
    const bool adapting = study != nullptr && (study->adaptive_step || study->recovery);
    std::optional<Config> step_config;
    if (adapting) step_config = config;
    std::size_t stagnant_generations = 0;
    std::vector<unsigned char> restart(study != nullptr && study->recovery ? population_size : 0, 0);
    std::vector<double> radius(adapting ? dimension : 0, 0.0);
    pending_positions.resize(update_batch_size);
    pending_velocities.resize(update_batch_size);
    pending_values.resize(update_batch_size);
    for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
        if (study != nullptr && expired(elapsed())) break;
        const auto first_particle = (iteration % batches_per_sweep) * update_batch_size;
        if (refinement) {
            refinement->plan(*study, particles, best_index, first_particle, iteration, batches_per_sweep);
        }
        std::size_t restart_count = 0;
        if (adapting && stagnant_generations >= 12U) {
            stagnant_generations = 0;
            ++study->triggers;
            if (study->adaptive_step) {
                step_config->dt = std::min(3.0 * config.dt, 1.25 * step_config->dt);
            }
            double dispersion = 0.0;
            for (std::size_t j = 0; j < dimension; ++j) {
                const double lower = config.lower_bounds[j];
                const double width = config.upper_bounds[j] - lower;
                double mean = 0.0;
                for (const auto& particle : particles) mean += (particle.position[j] - lower) / width;
                mean /= static_cast<double>(population_size);
                double variance = 0.0;
                for (const auto& particle : particles) {
                    const double deviation = (particle.position[j] - lower) / width - mean;
                    variance += deviation * deviation;
                }
                const double sigma = std::sqrt(variance / static_cast<double>(population_size));
                dispersion += sigma / static_cast<double>(dimension);
                radius[j] = width * std::clamp(2.0 * sigma, 0.01, 0.20);
            }
            study->mean_trigger_dispersion += (dispersion - study->mean_trigger_dispersion)
                / static_cast<double>(study->triggers);
            if (study->recovery && population_size > 1U) {
                std::vector<std::size_t> order(population_size);
                std::iota(order.begin(), order.end(), 0U);
                std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                    return particles[a].personal_best_value > particles[b].personal_best_value;
                });
                for (const auto i : order) {
                    if (i == best_index) continue;
                    restart[i] = 1;
                    if (++restart_count == std::max<std::size_t>(1U, population_size / 5U)) break;
                }
            }
        }
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

        std::vector<double> propagation_task_seconds(update_batch_size, 0.0);
        const auto propagation_start = Clock::now();
        const auto propagate_one = [&](const std::size_t slot) {
            const std::size_t i = first_particle + slot;
            const auto task_start = Clock::now();
            const std::vector<double>& attractor = config.topology == TopologyKind::GlobalBest
                ? global_best_snapshot : local_snapshot[i];
            const auto trial = refinement ? refinement->trials[slot] : RefinementTrial{};
            if (trial.kind != TrialKind::Swarm) {
                const auto& leader = particles[best_index].personal_best;
                auto& position = pending_positions[slot];
                position = leader;
                std::fill(pending_velocities[slot].begin(), pending_velocities[slot].end(), 0.0);
                if (trial.kind == TrialKind::Quadratic) {
                    position = refinement->quadratic_point;
                } else if (trial.kind == TrialKind::Local) {
                    const auto j = trial.axis;
                    const auto width = config.upper_bounds[j] - config.lower_bounds[j];
                    position[j] = std::clamp(position[j] + trial.sign * width * refinement->radius[j],
                                             config.lower_bounds[j], config.upper_bounds[j]);
                } else {
                    for (std::size_t j = 0; j < dimension; ++j) {
                        const auto lo = config.lower_bounds[j];
                        const auto hi = config.upper_bounds[j];
                        const auto u = uniform01(particles[i].random_engine);
                        position[j] = trial.kind == TrialKind::RestartGlobal ? lo + u * (hi - lo)
                            : std::clamp(leader[j] + (2.0 * u - 1.0) * 0.10 * (hi - lo), lo, hi);
                    }
                }
            } else if (restart_count != 0U && restart[i] != 0U) {
                auto& engine = particles[i].random_engine;
                for (std::size_t j = 0; j < dimension; ++j) {
                    const double offset = (2.0 * uniform01(engine) - 1.0) * radius[j];
                    pending_positions[slot][j] = std::clamp(attractor[j] + offset,
                        config.lower_bounds[j], config.upper_bounds[j]);
                    pending_velocities[slot][j] = 0.0;
                }
            } else {
                propagate_particle(
                    particles[i], attractor, adapting ? *step_config : config, iteration,
                    pending_positions[slot], pending_velocities[slot]);
            }
            propagation_task_seconds[slot] =
                std::chrono::duration<double>(Clock::now() - task_start).count();
        };
        if (particle_pool != nullptr) {
            particle_pool->parallel_for(update_batch_size, propagate_one);
        } else {
            for (std::size_t slot = 0; slot < update_batch_size; ++slot) {
                propagate_one(slot);
            }
        }
        result.propagation_seconds +=
            std::chrono::duration<double>(Clock::now() - propagation_start).count();
        for (const double seconds : propagation_task_seconds) {
            result.propagation_worker_seconds += seconds;
        }

        if (!accept_batch()) break;
        const double previous_best = result.best_value;
        if (refinement && study->local_search) {
            refinement->finish_local(config, pending_positions, pending_values,
                                     particles[best_index].personal_best, previous_best);
        }
        if (restart_count != 0U) study->recovered_particles += restart_count;

        for (std::size_t slot = 0; slot < update_batch_size; ++slot) {
            const std::size_t i = first_particle + slot;
            Particle& particle = particles[i];
            const auto kind = refinement ? refinement->trials[slot].kind : TrialKind::Swarm;
            const bool restarted = kind == TrialKind::RestartGlobal || kind == TrialKind::RestartNearLeader;
            if (kind == TrialKind::Local || kind == TrialKind::Quadratic) {
                const auto count = kind == TrialKind::Local ? 0U : 4U;
                ++study->refinement_counts[count];
                if (pending_values[slot] >= previous_best) continue;
                ++study->refinement_counts[count + 1U];
            }
            if (restarted) {
                ++study->refinement_counts[2];
                if (pending_values[slot] < previous_best) ++study->refinement_counts[3];
            }
            particle.position = pending_positions[slot];
            particle.velocity = pending_velocities[slot];
            particle.current_value = pending_values[slot];
            if (restart_count != 0U && restart[i] != 0U && particle.current_value < previous_best) {
                ++study->recovery_improvements;
            }
            if (restarted || (restart_count != 0U && restart[i] != 0U)
                || particle.current_value < particle.personal_best_value) {
                particle.personal_best = particle.position;
                particle.personal_best_value = particle.current_value;
                if (refinement) refinement->personal_progress[i] = iteration + 1U;
            }
            if (particle.personal_best_value < result.best_value) {
                result.best_value = particle.personal_best_value;
                result.best_position = particle.personal_best;
            }
        }
        if (restart_count != 0U) std::fill(restart.begin(), restart.end(), 0U);
        if (refinement && result.best_value < previous_best) refinement->last_progress = iteration + 1U;
        if (adapting) {
            if (result.best_value < previous_best) {
                stagnant_generations = 0;
                if (study->adaptive_step) step_config->dt = std::max(config.dt, 0.85 * step_config->dt);
            } else {
                ++stagnant_generations;
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
    return minimize_synchronous_impl(config, objective, serial_evaluator, 1U, nullptr, config.seed);
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
        result = minimize_synchronous_impl(config, objective, parallel_evaluator, worker_count, nullptr, config.seed);
    }
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

Result minimize_particle_parallel(
    const Config& config,
    const Objective& objective,
    const std::size_t worker_count) {
    return minimize_particle_parallel(config, objective, worker_count, config.seed);
}

Result minimize_particle_parallel(
    const Config& config, const Objective& objective,
    const std::size_t worker_count, const std::uint64_t graph_seed) {
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
        const auto parallel_evaluator = [&objective, &pool, &config](
            const std::vector<std::vector<double>>& positions,
            std::vector<double>& values,
            EvaluationTimings& timings) {
            if (config.study != nullptr && config.study->serial_objective) {
                evaluate_serial_batch(objective, positions, values, timings);
            } else {
                evaluate_parallel_batch(pool, objective, positions, values, timings);
            }
        };
        result = minimize_synchronous_impl(config, objective, parallel_evaluator, worker_count, &pool, graph_seed);
    }
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
    return result;
}

}  // namespace cpso
