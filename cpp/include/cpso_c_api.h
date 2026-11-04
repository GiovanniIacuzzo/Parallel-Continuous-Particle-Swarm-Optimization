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

enum CpsoTopology {
    CPSO_TOPOLOGY_INDEPENDENT = 0,
    CPSO_TOPOLOGY_STAR = 1,
    CPSO_TOPOLOGY_RING = 2
};

enum CpsoBenchmark {
    CPSO_BENCHMARK_SPHERE = 0,
    CPSO_BENCHMARK_RASTRIGIN = 1,
    CPSO_BENCHMARK_ROSENBROCK = 2,
    CPSO_BENCHMARK_SCHWEFEL = 3,
    CPSO_BENCHMARK_GRIEWANK = 4,
    CPSO_BENCHMARK_ACKLEY = 5,
    CPSO_BENCHMARK_SALOMON = 6
};

typedef struct CpsoConfig {
    uint32_t abi_version;
    int32_t dimension;
    int32_t population_size;       /* Global population across all workers. */
    int32_t iterations;             /* Number of state transitions. */
    int32_t workers;                /* OpenMP worker threads / islands. */
    int32_t sync_interval;           /* Communication period in iterations. */
    int32_t topology;                /* CpsoTopology. */
    int32_t benchmark;               /* CpsoBenchmark. */
    uint64_t seed;
    const double *lower_bounds;      /* Arrays of length dimension. */
    const double *upper_bounds;
    const double *optimum_position;  /* The minimizer in the user coordinates. */
    double mu_max;
    double mu_min;
    double cognitive;
    double social;
    double dt;
    double initial_velocity_fraction;
    double boundary_restitution;
} CpsoConfig;

typedef struct CpsoResult {
    double best_value;
    double *best_position;           /* Length dimension. */
    double *history;                 /* Length history_length. */
    size_t history_length;           /* Initial value + one per iteration. */
    double *worker_best_values;      /* Length worker_count. */
    size_t worker_count;
    uint64_t function_evaluations;
    double elapsed_seconds;
    double compute_seconds;
    double communication_seconds;
} CpsoResult;

CPSO_API int cpso_run(const CpsoConfig *config, CpsoResult *result,
                      char *error_message, size_t error_capacity);
CPSO_API void cpso_free_result(CpsoResult *result);
CPSO_API const char *cpso_version(void);
CPSO_API int cpso_openmp_enabled(void);
CPSO_API int cpso_max_threads(void);

#ifdef __cplusplus
}
#endif
#endif
