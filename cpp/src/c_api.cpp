#define CPSO_BUILDING_LIBRARY
#include "cpso/c_api.h"

#include "cpso/objectives.hpp"
#include "cpso/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kAbiVersion = 7U;
#ifndef CPSO_PROJECT_VERSION
#define CPSO_PROJECT_VERSION "0.7.4"
#endif
constexpr const char* kVersion = CPSO_PROJECT_VERSION;

void write_error(char* output, const std::size_t capacity, const char* message) noexcept {
    if (output == nullptr || capacity == 0U) {
        return;
    }
    const std::size_t length = std::min(capacity - 1U, std::strlen(message));
    std::memcpy(output, message, length);
    output[length] = '\0';
}

cpso::TopologyKind to_topology(const int id) {
    switch (id) {
        case CPSO_TOPOLOGY_GLOBAL_BEST: return cpso::TopologyKind::GlobalBest;
        case CPSO_TOPOLOGY_RING: return cpso::TopologyKind::Ring;
        case CPSO_TOPOLOGY_GRID_VON_NEUMANN: return cpso::TopologyKind::GridVonNeumann;
        case CPSO_TOPOLOGY_WHEEL: return cpso::TopologyKind::Wheel;
        case CPSO_TOPOLOGY_RANDOM_CONNECTED: return cpso::TopologyKind::RandomConnected;
        case CPSO_TOPOLOGY_SMALL_WORLD: return cpso::TopologyKind::SmallWorld;
        case CPSO_TOPOLOGY_SCALE_FREE: return cpso::TopologyKind::ScaleFree;
        default: throw std::invalid_argument("Unknown topology_id");
    }
}

cpso::Benchmark to_benchmark(const int id) {
    switch (id) {
        case CPSO_SPHERE: return cpso::Benchmark::Sphere;
        case CPSO_RASTRIGIN: return cpso::Benchmark::Rastrigin;
        case CPSO_ROSENBROCK: return cpso::Benchmark::Rosenbrock;
        case CPSO_ACKLEY: return cpso::Benchmark::Ackley;
        case CPSO_GRIEWANK: return cpso::Benchmark::Griewank;
        case CPSO_SALOMON: return cpso::Benchmark::Salomon;
        default: throw std::invalid_argument("Unknown benchmark_id");
    }
}

cpso::MigrationTopology to_migration_topology(const int id) {
    switch (id) {
        case CPSO_MIGRATION_RING: return cpso::MigrationTopology::Ring;
        case CPSO_MIGRATION_FULLY_CONNECTED: return cpso::MigrationTopology::FullyConnected;
        default: throw std::invalid_argument("Unknown migration_topology_id");
    }
}

void copy_result(
    const cpso::Result& source,
    CpsoCResult* destination,
    double* best_position_out,
    double* history_out,
    std::uint64_t* history_evaluations_out,
    const std::size_t history_capacity) {
    if (source.best_history.size() > history_capacity
        || source.history_evaluations.size() != source.best_history.size()) {
        throw std::runtime_error("Internal history-size contract violated");
    }
    std::copy(source.best_position.begin(), source.best_position.end(), best_position_out);
    std::copy(source.best_history.begin(), source.best_history.end(), history_out);
    if (history_evaluations_out != nullptr) {
        std::copy(
            source.history_evaluations.begin(), source.history_evaluations.end(),
            history_evaluations_out);
    }

    destination->best_value = source.best_value;
    destination->function_evaluations = source.function_evaluations;
    destination->iterations_completed = static_cast<int>(source.iterations_completed);
    destination->history_size = source.best_history.size();
    destination->elapsed_seconds = source.elapsed_seconds;
    destination->evaluation_wall_seconds = source.evaluation_wall_seconds;
    destination->propagation_seconds = source.propagation_seconds;
    destination->topology_setup_seconds = source.topology_setup_seconds;
    destination->social_selection_seconds = source.social_selection_seconds;
    destination->worker_count = source.worker_count;
    destination->initialization_seconds = source.initialization_seconds;
    destination->propagation_worker_seconds = source.propagation_worker_seconds;
    destination->evaluation_worker_seconds = source.evaluation_worker_seconds;
    destination->migration_seconds = source.migration_seconds;
    destination->async_update_wall_seconds = source.async_update_wall_seconds;
    destination->migrations_performed = source.migrations_performed;
    destination->island_count = source.island_count;
}

int run_impl(
    const CpsoCConfig* config,
    const CpsoCExecutionConfig* execution,
    CpsoCResult* c_result,
    double* best_position_out,
    double* history_out,
    std::uint64_t* history_evaluations_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity,
    const CpsoObjectiveCallback custom_objective = nullptr,
    void* custom_objective_context = nullptr) {
    try {
        if (config == nullptr || execution == nullptr || c_result == nullptr
            || best_position_out == nullptr || history_out == nullptr) {
            throw std::invalid_argument("C API received a null required pointer");
        }
        if (config->abi_version != kAbiVersion || execution->abi_version != kAbiVersion) {
            throw std::invalid_argument("Unsupported C API ABI version");
        }
        if (config->dimension < 1 || config->population_size < 1 || config->iterations < 0) {
            throw std::invalid_argument("dimension/population_size must be positive and iterations non-negative");
        }
        if (config->grid_rows < 0 || config->grid_columns < 0) {
            throw std::invalid_argument("grid_rows and grid_columns must be non-negative");
        }
        if (config->objective_work_iterations > 100'000'000ULL) {
            throw std::invalid_argument("objective_work_iterations exceeds 100000000");
        }
        if (config->lower_bounds == nullptr || config->upper_bounds == nullptr) {
            throw std::invalid_argument("Bounds pointers must not be null");
        }
        if (history_evaluations_out == nullptr) {
            throw std::invalid_argument("history_evaluations_out must not be null");
        }
        const std::size_t dimension = static_cast<std::size_t>(config->dimension);
        const std::size_t population_size = static_cast<std::size_t>(config->population_size);
        const std::size_t iterations = static_cast<std::size_t>(config->iterations);
        std::size_t expected_history = iterations + 1U;
        if (execution->algorithm_id == CPSO_ALGORITHM_ASYNCHRONOUS) {
            if (iterations > 0U
                && population_size > (std::numeric_limits<std::size_t>::max() - 1U) / iterations) {
                throw std::invalid_argument("Asynchronous history capacity overflows size_t");
            }
            expected_history = population_size * iterations + 1U;
        }
        if (history_capacity < expected_history) {
            throw std::invalid_argument("history_out capacity is too small for the selected architecture");
        }

        if (config->topology_degree < 1) {
            throw std::invalid_argument("topology_degree must be positive");
        }
        if (!std::isfinite(config->topology_rewiring_probability)
            || config->topology_rewiring_probability < 0.0
            || config->topology_rewiring_probability > 1.0) {
            throw std::invalid_argument("topology_rewiring_probability must lie in [0,1]");
        }

        cpso::Config cpp_config;
        cpp_config.lower_bounds.assign(config->lower_bounds, config->lower_bounds + dimension);
        cpp_config.upper_bounds.assign(config->upper_bounds, config->upper_bounds + dimension);
        cpp_config.population_size = population_size;
        cpp_config.iterations = iterations;
        cpp_config.seed = config->seed;
        cpp_config.mu_max = config->mu_max;
        cpp_config.mu_min = config->mu_min;
        cpp_config.cognitive = config->cognitive;
        cpp_config.social = config->social;
        cpp_config.dt = config->dt;
        cpp_config.initial_velocity_fraction = config->initial_velocity_fraction;
        cpp_config.boundary_restitution = config->boundary_restitution;
        cpp_config.topology = to_topology(config->topology_id);
        cpp_config.grid_rows = static_cast<std::size_t>(config->grid_rows);
        cpp_config.grid_columns = static_cast<std::size_t>(config->grid_columns);
        cpp_config.topology_degree = static_cast<std::size_t>(config->topology_degree);
        cpp_config.topology_rewiring_probability = config->topology_rewiring_probability;

        cpso::Objective objective;
        if (custom_objective != nullptr) {
            if (execution->algorithm_id != CPSO_ALGORITHM_SERIAL) {
                throw std::invalid_argument(
                    "Custom objective callbacks are supported only in serial mode");
            }
            if (config->objective_work_iterations != 0U) {
                throw std::invalid_argument(
                    "objective_work_iterations is not valid with a custom objective");
            }
            objective = [custom_objective, custom_objective_context](
                const std::vector<double>& position) {
                return custom_objective(
                    position.data(), position.size(), custom_objective_context);
            };
        } else if (config->benchmark_id == CPSO_SPHERE_WORK) {
            objective = [work_iterations = config->objective_work_iterations](
                const std::vector<double>& position) {
                return cpso::evaluate_synthetic_sphere_work(position, work_iterations);
            };
        } else {
            if (config->objective_work_iterations != 0U) {
                throw std::invalid_argument(
                    "objective_work_iterations is only valid for sphere_work");
            }
            const cpso::Benchmark benchmark = to_benchmark(config->benchmark_id);
            objective = [benchmark](const std::vector<double>& position) {
                return cpso::evaluate_benchmark(benchmark, position);
            };
        }

        cpso::Result result;
        switch (execution->algorithm_id) {
            case CPSO_ALGORITHM_SERIAL:
                result = cpso::minimize_serial(cpp_config, objective);
                break;
            case CPSO_ALGORITHM_MASTER_SLAVE:
                result = cpso::minimize_master_slave(
                    cpp_config, objective, execution->worker_count);
                break;
            case CPSO_ALGORITHM_PARTICLE_PARALLEL:
                result = cpso::minimize_particle_parallel(
                    cpp_config, objective, execution->worker_count);
                break;
            case CPSO_ALGORITHM_ASYNCHRONOUS:
                result = cpso::minimize_asynchronous(
                    cpp_config, objective, execution->worker_count);
                break;
            case CPSO_ALGORITHM_ISLANDS: {
                cpso::IslandConfig island_config;
                island_config.island_count = execution->island_count;
                island_config.worker_count = execution->worker_count;
                island_config.migration_interval = execution->migration_interval;
                island_config.migration_topology = to_migration_topology(
                    execution->migration_topology_id);
                result = cpso::minimize_islands(cpp_config, objective, island_config);
                break;
            }
            default:
                throw std::invalid_argument("Unknown algorithm_id");
        }

        copy_result(result, c_result, best_position_out, history_out,
                    history_evaluations_out, history_capacity);
        if (error_message != nullptr && error_capacity > 0U) {
            error_message[0] = '\0';
        }
        return 0;
    } catch (const std::exception& exception) {
        write_error(error_message, error_capacity, exception.what());
        return -1;
    } catch (...) {
        write_error(error_message, error_capacity, "Unknown native exception");
        return -1;
    }
}

CpsoCExecutionConfig legacy_execution(const int algorithm_id, const std::size_t workers) {
    CpsoCExecutionConfig execution{};
    execution.abi_version = kAbiVersion;
    execution.algorithm_id = algorithm_id;
    execution.worker_count = workers;
    execution.island_count = 0U;
    execution.migration_interval = 0U;
    execution.migration_topology_id = CPSO_MIGRATION_RING;
    return execution;
}
}  // namespace

extern "C" CPSO_API std::uint32_t cpso_abi_version(void) {
    return kAbiVersion;
}

extern "C" CPSO_API const char* cpso_version(void) {
    return kVersion;
}

extern "C" CPSO_API const char* cpso_build_info(void) {
#ifndef CPSO_COMPILER_ID
#define CPSO_COMPILER_ID "unknown"
#endif
#ifndef CPSO_COMPILER_VERSION
#define CPSO_COMPILER_VERSION "unknown"
#endif
#ifndef CPSO_BUILD_TYPE
#define CPSO_BUILD_TYPE "unspecified"
#endif
#ifndef CPSO_CXX_FLAGS
#define CPSO_CXX_FLAGS "unspecified"
#endif
#ifndef CPSO_TARGET_PROCESSOR
#define CPSO_TARGET_PROCESSOR "unspecified"
#endif
    static const std::string info =
        std::string("compiler=") + CPSO_COMPILER_ID
        + ";compiler_version=" + CPSO_COMPILER_VERSION
        + ";build_type=" + CPSO_BUILD_TYPE
        + ";cxx_standard=20;target_processor=" + CPSO_TARGET_PROCESSOR
        + ";cxx_flags=" + CPSO_CXX_FLAGS
        + ";abi=" + std::to_string(kAbiVersion);
    return info.c_str();
}

extern "C" CPSO_API int cpso_run_serial(
    const CpsoCConfig* config,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity) {
    const CpsoCExecutionConfig execution = legacy_execution(CPSO_ALGORITHM_SERIAL, 1U);
    std::vector<std::uint64_t> unused_history_evaluations(history_capacity, 0U);
    return run_impl(config, &execution, result, best_position_out, history_out,
                    unused_history_evaluations.data(), history_capacity,
                    error_message, error_capacity);
}

extern "C" CPSO_API int cpso_run_master_slave(
    const CpsoCConfig* config,
    const std::size_t worker_count,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity) {
    const CpsoCExecutionConfig execution = legacy_execution(CPSO_ALGORITHM_MASTER_SLAVE, worker_count);
    std::vector<std::uint64_t> unused_history_evaluations(history_capacity, 0U);
    return run_impl(config, &execution, result, best_position_out, history_out,
                    unused_history_evaluations.data(), history_capacity,
                    error_message, error_capacity);
}

extern "C" CPSO_API int cpso_run_custom_serial(
    const CpsoCConfig* config,
    const CpsoObjectiveCallback objective_callback,
    void* objective_context,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    std::uint64_t* history_evaluations_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity) {
    const CpsoCExecutionConfig execution = legacy_execution(CPSO_ALGORITHM_SERIAL, 1U);
    if (objective_callback == nullptr) {
        write_error(error_message, error_capacity, "Custom objective callback must not be null");
        return -1;
    }
    return run_impl(
        config, &execution, result, best_position_out, history_out,
        history_evaluations_out, history_capacity, error_message, error_capacity,
        objective_callback, objective_context);
}

extern "C" CPSO_API int cpso_run_architecture(
    const CpsoCConfig* config,
    const CpsoCExecutionConfig* execution,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    std::uint64_t* history_evaluations_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity) {
    return run_impl(config, execution, result, best_position_out, history_out,
                    history_evaluations_out, history_capacity,
                    error_message, error_capacity);
}
