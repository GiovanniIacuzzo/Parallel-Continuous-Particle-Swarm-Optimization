#ifndef CPSO_C_API_H
#define CPSO_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(CPSO_BUILDING_LIBRARY)
#    define CPSO_API __declspec(dllexport)
#  else
#    define CPSO_API __declspec(dllimport)
#  endif
#else
#  define CPSO_API __attribute__((visibility("default")))
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
    CPSO_SPHERE_WORK = 7,
    CPSO_INVERSE_DIFFUSION = 8
};

enum CpsoTopologyId {
    CPSO_TOPOLOGY_GLOBAL_BEST = 0,
    CPSO_TOPOLOGY_RING = 1,
    CPSO_TOPOLOGY_GRID_VON_NEUMANN = 2,
    CPSO_TOPOLOGY_WHEEL = 3,
    CPSO_TOPOLOGY_RANDOM_CONNECTED = 4,
    CPSO_TOPOLOGY_SMALL_WORLD = 5,
    CPSO_TOPOLOGY_SCALE_FREE = 6
};

enum CpsoAlgorithmId {
    CPSO_ALGORITHM_SERIAL = 0,
    CPSO_ALGORITHM_MASTER_SLAVE = 1,
    CPSO_ALGORITHM_PARTICLE_PARALLEL = 2,
    CPSO_ALGORITHM_ASYNCHRONOUS = 3,
    CPSO_ALGORITHM_ISLANDS = 4
};

enum CpsoMigrationTopologyId {
    CPSO_MIGRATION_RING = 0,
    CPSO_MIGRATION_FULLY_CONNECTED = 1
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
    int topology_id;
    int grid_rows;
    int grid_columns;
    uint64_t objective_work_iterations;
    // ABI v7 graph-topology parameters. topology_degree means expected degree
    // (random), ring degree (small-world), or 2*m (scale-free).
    int topology_degree;
    double topology_rewiring_probability;
} CpsoCConfig;

typedef struct CpsoCExecutionConfig {
    uint32_t abi_version;
    int algorithm_id;
    size_t worker_count;
    size_t island_count;
    size_t migration_interval;
    int migration_topology_id;
} CpsoCExecutionConfig;

// Custom scalar objective callback. This callback is intentionally supported
// only by cpso_run_custom_serial: Python/foreign callbacks are not assumed to
// be thread-safe and must not be invoked concurrently from native workers.
typedef double (*CpsoObjectiveCallback)(
    const double* position, size_t dimension, void* user_data);

typedef struct CpsoCResult {
    double best_value;
    uint64_t function_evaluations;
    int iterations_completed;
    size_t history_size;
    double elapsed_seconds;
    double evaluation_wall_seconds;
    double propagation_seconds;
    double topology_setup_seconds;
    double social_selection_seconds;
    size_t worker_count;
    double initialization_seconds;
    double propagation_worker_seconds;
    double evaluation_worker_seconds;
    double migration_seconds;
    double async_update_wall_seconds;
    uint64_t migrations_performed;
    size_t island_count;
} CpsoCResult;

CPSO_API uint32_t cpso_abi_version(void);
CPSO_API const char* cpso_version(void);
CPSO_API const char* cpso_build_info(void);

// Legacy convenience entry points retained for serial/master--slave clients.
// ABI v5 extends CpsoCResult with phase and migration counters.
CPSO_API int cpso_run_serial(
    const CpsoCConfig* config,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

CPSO_API int cpso_run_master_slave(
    const CpsoCConfig* config,
    size_t worker_count,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

// Run the serial CPSO solver against a caller-owned objective function.
// The callback must return a finite scalar value and must not throw across the
// C ABI. A non-finite return value aborts the optimization with an error.
CPSO_API int cpso_run_custom_serial(
    const CpsoCConfig* config,
    CpsoObjectiveCallback objective_callback,
    void* objective_context,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    uint64_t* history_evaluations_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

// Unified execution entry point. For synchronous algorithms history_capacity
// must be >= iterations+1. For asynchronous CPSO it must be >=
// population_size*iterations+1 because history is recorded after every
// committed particle update. history_evaluations_out aligns each history value
// with the number of objective evaluations completed at that observation.
CPSO_API int cpso_run_architecture(
    const CpsoCConfig* config,
    const CpsoCExecutionConfig* execution,
    CpsoCResult* result,
    double* best_position_out,
    double* history_out,
    uint64_t* history_evaluations_out,
    size_t history_capacity,
    char* error_message,
    size_t error_capacity);

// Deterministically reconstruct native initial x/v for subspace diagnostics.
CPSO_API int cpso_experiment_initial_state(uint64_t, size_t, size_t,
    const double*, const double*, double, double*, double*, char*, size_t);

// Adjacency includes self; caller supplies n*n bytes (n <= 4096).
CPSO_API int cpso_experiment_graph(int, size_t, size_t, size_t, uint64_t,
                                  size_t, double, unsigned char*, char*, size_t);

// Opt-in asynchronous comparison: flags 0=original, 16=adaptive dt+recovery.
// time_limit_seconds=0 means FE cap only; summary contains final best/FE.
// The maximum FE is population_size*(iterations+1).
CPSO_API int cpso_run_research(
    const CpsoCConfig*, const CpsoCExecutionConfig*, int flags, double time_limit_seconds,
    CpsoCResult*, double*, double*, uint64_t*, size_t,
    char*, size_t);

// Batch timestamps align with the synchronous history. Options and diagnostics
// belong to the caller and are valid for the duration of a single run.
typedef struct {
    double seconds;                  // 0 means FE budget only
    size_t schedule_iterations;      // 0 uses the planned FE budget
    int adapt_step;
    int recovery;
    uint64_t discarded_evaluations;
    uint64_t triggers;
    uint64_t recovered_particles;
    uint64_t recovery_improvements;
    double mean_trigger_dispersion;
} CpsoSynchronousStudy;

CPSO_API int cpso_run_synchronous_study(
    const CpsoCConfig*, const CpsoCExecutionConfig*, int flags, uint64_t graph_seed,
    CpsoObjectiveCallback, CpsoSynchronousStudy*, CpsoCResult*, double*,
    double*, uint64_t*, double* batch_seconds, size_t, char*, size_t);

// Persistent pool for baseline objective batches, with the same native
// evaluator used by the synchronous CPSO. Input is row-major.
CPSO_API void* cpso_create_batch_evaluator(int benchmark, uint64_t work, size_t workers, char*, size_t);
CPSO_API int cpso_evaluate_batch(void*, const double*, size_t rows, size_t dimension,
                                double*, char*, size_t);
CPSO_API void cpso_destroy_batch_evaluator(void*);

// Additive research entry point; existing ABI structures are unchanged.
// flags: bit 0 discrete, bit 1 coordinate randomness, bit 2 separate graph seed,
// bit 3 disable boundary map (unconstrained geometric diagnostics only).
CPSO_API int cpso_run_experiment(
    const CpsoCConfig*, const CpsoCExecutionConfig*, int flags, uint64_t graph_seed,
    CpsoObjectiveCallback, void*, CpsoCResult*, double*, double*, uint64_t*,
    size_t, char*, size_t);

#ifdef __cplusplus
}
#endif
#endif
