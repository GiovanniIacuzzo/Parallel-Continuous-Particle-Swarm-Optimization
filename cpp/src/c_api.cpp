#define CPSO_BUILDING_LIBRARY
#include "cpso/c_api.h"

#include "cpso/objectives.hpp"
#include "cpso/optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kAbiVersion = 3U;
constexpr const char* kVersion = "0.3.0";

void write_error(char* output, const std::size_t capacity, const char* message) noexcept {
    if (output == nullptr || capacity == 0U) {
        return;
    }
    const std::size_t length = std::min(capacity - 1U, std::strlen(message));
    std::memcpy(output, message, length);
    output[length] = '\0';
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

int run_impl(
    const CpsoCConfig* config,
    const std::size_t worker_count,
    const bool master_slave,
    CpsoCResult* c_result,
    double* best_position_out,
    double* history_out,
    const std::size_t history_capacity,
    char* error_message,
    const std::size_t error_capacity) {
    try {
        if (config == nullptr || c_result == nullptr || best_position_out == nullptr || history_out == nullptr) {
            throw std::invalid_argument("C API received a null required pointer");
        }
        if (config->abi_version != kAbiVersion) {
            throw std::invalid_argument("Unsupported C API ABI version");
        }
        if (config->dimension < 1 || config->population_size < 1 || config->iterations < 0) {
            throw std::invalid_argument("dimension/population_size must be positive and iterations non-negative");
        }
        if (master_slave && worker_count == 0U) {
            throw std::invalid_argument("worker_count must be >= 1");
        }
        if (config->objective_work_iterations > 100'000'000ULL) {
            throw std::invalid_argument("objective_work_iterations exceeds 100000000");
        }
        if (config->lower_bounds == nullptr || config->upper_bounds == nullptr) {
            throw std::invalid_argument("Bounds pointers must not be null");
        }
        const std::size_t dimension = static_cast<std::size_t>(config->dimension);
        const std::size_t expected_history = static_cast<std::size_t>(config->iterations) + 1U;
        if (history_capacity < expected_history) {
            throw std::invalid_argument("history_out capacity must be at least iterations + 1");
        }

        cpso::Config cpp_config;
        cpp_config.lower_bounds.assign(config->lower_bounds, config->lower_bounds + dimension);
        cpp_config.upper_bounds.assign(config->upper_bounds, config->upper_bounds + dimension);
        cpp_config.population_size = static_cast<std::size_t>(config->population_size);
        cpp_config.iterations = static_cast<std::size_t>(config->iterations);
        cpp_config.seed = config->seed;
        cpp_config.mu_max = config->mu_max;
        cpp_config.mu_min = config->mu_min;
        cpp_config.cognitive = config->cognitive;
        cpp_config.social = config->social;
        cpp_config.dt = config->dt;
        cpp_config.initial_velocity_fraction = config->initial_velocity_fraction;
        cpp_config.boundary_restitution = config->boundary_restitution;

        cpso::Objective objective;
        if (config->benchmark_id == CPSO_SPHERE_WORK) {
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
        const cpso::Result result = master_slave
            ? cpso::minimize_master_slave(cpp_config, objective, worker_count)
            : cpso::minimize_serial(cpp_config, objective);

        std::copy(result.best_position.begin(), result.best_position.end(), best_position_out);
        std::copy(result.best_history.begin(), result.best_history.end(), history_out);
        c_result->best_value = result.best_value;
        c_result->function_evaluations = result.function_evaluations;
        c_result->iterations_completed = static_cast<int>(result.iterations_completed);
        c_result->history_size = result.best_history.size();
        c_result->elapsed_seconds = result.elapsed_seconds;
        c_result->evaluation_wall_seconds = result.evaluation_wall_seconds;
        c_result->propagation_seconds = result.propagation_seconds;
        c_result->worker_count = result.worker_count;
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
    static const std::string info =
        std::string("compiler=") + CPSO_COMPILER_ID
        + ";compiler_version=" + CPSO_COMPILER_VERSION
        + ";build_type=" + CPSO_BUILD_TYPE
        + ";cxx_standard=20;abi=" + std::to_string(kAbiVersion);
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
    return run_impl(config, 1U, false, result, best_position_out, history_out,
                    history_capacity, error_message, error_capacity);
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
    return run_impl(config, worker_count, true, result, best_position_out, history_out,
                    history_capacity, error_message, error_capacity);
}
