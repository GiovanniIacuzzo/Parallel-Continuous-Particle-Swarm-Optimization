#ifndef CPSO_ENGINE_HPP
#define CPSO_ENGINE_HPP

#include "cpso_c_api.h"
#include <vector>

namespace cpso {
struct EngineResult {
    double best_value{};
    std::vector<double> best_position;
    std::vector<double> history;
    std::vector<double> worker_best_values;
    uint64_t function_evaluations{};
    double elapsed_seconds{};
    double compute_seconds{};
    double communication_seconds{};
};

EngineResult run(const CpsoConfig &config);
void exact_oscillator_step(double x, double v, double equilibrium,
                           double omega2, double mu, double dt,
                           double &x_next, double &v_next);
double evaluate_benchmark(int benchmark, const std::vector<double> &x,
                          const std::vector<double> &optimum);
}
#endif
