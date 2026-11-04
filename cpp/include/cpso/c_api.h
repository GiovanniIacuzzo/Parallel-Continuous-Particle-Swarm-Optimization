#ifndef CPSO_C_API_H
#define CPSO_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(CPSO_BUILDING_LIBRARY)
    #define CPSO_API __declspec(dllexport)
  #else
    #define CPSO_API __declspec(dllimport)
  #endif
#else
  #define CPSO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum CpsoBenchmarkId {
    CPSO_SPHERE = 1,
    CPSO_RASTRIGIN = 2,
    CPSO_ROSENBROCK = 3,
    CPSO_ACKLEY = 4,
    CPSO_GRIEWANK = 5,
    CPSO_SALOMON = 6,
    CPSO_SPHERE_WORK = 7
};

typedef struct CpsoCConfig {
    uint32_t abi_version;
    int benchmark_id;
    int dimension;
    int population_size;
    int iterations;
    uint64_t seed;
    const double* lower_bounds;
    const double* upper_bounds;
    double mu_max;
    double mu_min;
    double cognitive;
    double social;
    double dt;
    double initial_velocity_fraction;
    double boundary_restitution;
    uint64_t objective_work_iterations;
} CpsoCConfig;

typedef struct CpsoCResult {
    double best_value;
    uint64_t function_evaluations;
    int iterations_completed;
    size_t history_size;
    double elapsed_seconds;
    double evaluation_wall_seconds;
    double propagation_seconds;
    size_t worker_count;
} CpsoCResult;

CPSO_API uint32_t cpso_abi_version(void);
CPSO_API const char* cpso_version(void);
CPSO_API const char* cpso_build_info(void);

// Returns 0 on success and -1 on error. On success, best_position_out must
// provide at least dimension values and history_out at least iterations+1.
// On error, error_message receives a null-terminated diagnostic when possible.
CPSO_API int cpso_run_serial(
    const CpsoCConfig* config,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

// Synchronous master--slave mode: only objective evaluations are executed on
// worker threads. worker_count must be >= 1. Built-in objectives, including
// the explicitly synthetic sphere_work arithmetic microbenchmark, are safe
// for concurrent evaluation. ABI v3 adds objective_work_iterations.
CPSO_API int cpso_run_master_slave(
    const CpsoCConfig* config,
    size_t worker_count,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
