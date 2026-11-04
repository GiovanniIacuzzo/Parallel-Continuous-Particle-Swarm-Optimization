#define CPSO_BUILDING_LIBRARY
#include "cpso_c_api.h"
#include "cpso_engine.hpp"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <exception>
#include <new>
#include <string>

#ifdef CPSO_WITH_OPENMP
  #include <omp.h>
#endif

extern "C" {
int cpso_run(const CpsoConfig *config, CpsoResult *result,
             char *error_message, size_t error_capacity) {
    if (result == nullptr || config == nullptr) {
        if (error_message != nullptr && error_capacity > 0)
            std::snprintf(error_message, error_capacity, "config and result pointers must not be null.");
        return 1;
    }
    std::memset(result, 0, sizeof(CpsoResult));
    try {
        const cpso::EngineResult native = cpso::run(*config);
        result->best_value = native.best_value;
        result->history_length = native.history.size();
        result->worker_count = native.worker_best_values.size();
        result->function_evaluations = native.function_evaluations;
        result->elapsed_seconds = native.elapsed_seconds;
        result->compute_seconds = native.compute_seconds;
        result->communication_seconds = native.communication_seconds;

        result->best_position = new double[native.best_position.size()];
        std::copy(native.best_position.begin(), native.best_position.end(), result->best_position);
        result->history = new double[native.history.size()];
        std::copy(native.history.begin(), native.history.end(), result->history);
        result->worker_best_values = new double[native.worker_best_values.size()];
        std::copy(native.worker_best_values.begin(), native.worker_best_values.end(), result->worker_best_values);
        if (error_message != nullptr && error_capacity > 0) error_message[0] = '\0';
        return 0;
    } catch (const std::exception &e) {
        cpso_free_result(result);
        if (error_message != nullptr && error_capacity > 0)
            std::snprintf(error_message, error_capacity, "%s", e.what());
        return 2;
    } catch (...) {
        cpso_free_result(result);
        if (error_message != nullptr && error_capacity > 0)
            std::snprintf(error_message, error_capacity, "Unknown C++ exception.");
        return 3;
    }
}

void cpso_free_result(CpsoResult *result) {
    if (result == nullptr) return;
    delete[] result->best_position;
    delete[] result->history;
    delete[] result->worker_best_values;
    std::memset(result, 0, sizeof(CpsoResult));
}

const char *cpso_version(void) { return "0.1.0"; }

int cpso_openmp_enabled(void) {
#ifdef CPSO_WITH_OPENMP
    return 1;
#else
    return 0;
#endif
}

int cpso_max_threads(void) {
#ifdef CPSO_WITH_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
}
