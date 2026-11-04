#include "cpso_engine.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool cond, const char *message) {
    if (!cond) throw std::runtime_error(message);
}

CpsoConfig make_config(int workers) {
    static std::vector<double> lb(8, -5.12), ub(8, 5.12), optimum(8, 0.0);
    CpsoConfig c{};
    c.abi_version = 1;
    c.dimension = 8;
    c.population_size = 64;
    c.iterations = 80;
    c.workers = workers;
    c.sync_interval = 1;
    c.topology = CPSO_TOPOLOGY_STAR;
    c.benchmark = CPSO_BENCHMARK_RASTRIGIN;
    c.seed = 991827;
    c.lower_bounds = lb.data();
    c.upper_bounds = ub.data();
    c.optimum_position = optimum.data();
    c.mu_max = 0.9;
    c.mu_min = 0.4;
    c.cognitive = 2.05;
    c.social = 2.05;
    c.dt = 0.7;
    c.initial_velocity_fraction = 0.1;
    c.boundary_restitution = 0.5;
    return c;
}
}

int main() {
    try {
        // Basic benchmark sanity at known minimizers under the shifted-coordinate convention.
        std::vector<double> opt(4, 0.0);
        std::vector<double> zero(4, 0.0);
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_SPHERE, zero, opt)) < 1e-14, "Sphere optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_RASTRIGIN, zero, opt)) < 1e-12, "Rastrigin optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_ROSENBROCK, zero, opt)) < 1e-12, "Rosenbrock optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_SCHWEFEL, zero, opt)) < 1e-10, "Schwefel shifted optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_GRIEWANK, zero, opt)) < 1e-12, "Griewank optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_ACKLEY, zero, opt)) < 1e-12, "Ackley optimum");
        require(std::abs(cpso::evaluate_benchmark(CPSO_BENCHMARK_SALOMON, zero, opt)) < 1e-12, "Salomon optimum");

        // The closed-form propagator must satisfy the semigroup property for
        // frozen coefficients in underdamped, critical and overdamped regimes.
        for (const auto &case_data : std::vector<std::pair<double, double>>{
                 {0.9, 2.0},   // underdamped: alpha=0.05, omega^2=2
                 {0.4, 0.09},  // critical: alpha=0.3, omega^2=0.09
                 {0.4, 0.02}}) // overdamped: alpha=0.3, omega^2=0.02
        {
            const double mu = case_data.first;
            const double omega2 = case_data.second;
            double x_full = 0.0, v_full = 0.0;
            double x_half = 0.0, v_half = 0.0;
            double x_twice = 0.0, v_twice = 0.0;
            cpso::exact_oscillator_step(1.2, -0.35, 0.4, omega2, mu, 0.8, x_full, v_full);
            cpso::exact_oscillator_step(1.2, -0.35, 0.4, omega2, mu, 0.4, x_half, v_half);
            cpso::exact_oscillator_step(x_half, v_half, 0.4, omega2, mu, 0.4, x_twice, v_twice);
            require(std::abs(x_full - x_twice) < 2e-12, "Frozen ODE flow semigroup property (position)");
            require(std::abs(v_full - v_twice) < 2e-12, "Frozen ODE flow semigroup property (velocity)");
        }

        // Fixed-seed repeatability.
        CpsoConfig c1 = make_config(1);
        CpsoConfig c2 = make_config(1);
        const auto r1 = cpso::run(c1);
        const auto r2 = cpso::run(c2);
        require(r1.history == r2.history, "Fixed-seed repeatability");
        require(r1.best_position == r2.best_position, "Fixed-seed position repeatability");
        require(r1.function_evaluations == 64ULL * 81ULL, "Correct FE accounting");
        require(r1.history.size() == 81U, "History length");
        for (std::size_t i = 1; i < r1.history.size(); ++i)
            require(r1.history[i] <= r1.history[i - 1] + 1e-13, "Best-so-far history monotonicity");

        // Star topology is algorithmically invariant to the worker partition when sync=1.
        CpsoConfig c4 = make_config(4);
        const auto r4 = cpso::run(c4);
        require(r1.history == r4.history, "Star result must not depend on worker count");
        require(r1.best_position == r4.best_position, "Star position must not depend on worker count");

        std::cout << "All C++ tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "TEST FAILURE: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
