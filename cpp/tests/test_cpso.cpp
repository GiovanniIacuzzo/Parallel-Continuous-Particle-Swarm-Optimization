#include "cpso/analytic_propagator.hpp"
#include "cpso/objectives.hpp"
#include "cpso/optimizer.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " (actual=" + std::to_string(actual)
                                 + ", expected=" + std::to_string(expected) + ")");
    }
}

cpso::OscillatorState rk4_reference(
    const double x0, const double v0, const double equilibrium,
    const double omega2, const double mu, const double dt, const int steps) {
    const double alpha = 0.5 * (1.0 - mu);
    const double h = dt / static_cast<double>(steps);
    double u = x0 - equilibrium;
    double v = v0;
    const auto rhs = [alpha, omega2](const double position, const double velocity) {
        return std::pair<double, double>{velocity, -2.0 * alpha * velocity - omega2 * position};
    };
    for (int i = 0; i < steps; ++i) {
        const auto k1 = rhs(u, v);
        const auto k2 = rhs(u + 0.5 * h * k1.first, v + 0.5 * h * k1.second);
        const auto k3 = rhs(u + 0.5 * h * k2.first, v + 0.5 * h * k2.second);
        const auto k4 = rhs(u + h * k3.first, v + h * k3.second);
        u += (h / 6.0) * (k1.first + 2.0 * k2.first + 2.0 * k3.first + k4.first);
        v += (h / 6.0) * (k1.second + 2.0 * k2.second + 2.0 * k3.second + k4.second);
    }
    return {equilibrium + u, v};
}

cpso::Config test_config() {
    cpso::Config config;
    config.lower_bounds = {-5.12, -5.12, -5.12, -5.12};
    config.upper_bounds = {5.12, 5.12, 5.12, 5.12};
    config.population_size = 32;
    config.iterations = 40;
    config.seed = 20261104;
    config.mu_max = 0.9;
    config.mu_min = 0.4;
    config.cognitive = 2.05;
    config.social = 2.05;
    config.dt = 0.7;
    config.initial_velocity_fraction = 0.1;
    config.boundary_restitution = 0.5;
    return config;
}

void test_propagator_against_rk4() {
    // Under-, critically-, and over-damped regimes.
    const std::vector<std::pair<double, double>> cases{
        {0.9, 2.0},   // alpha=0.05, discriminant < 0
        {0.4, 0.09},  // alpha=0.3, discriminant = 0
        {0.4, 0.02}   // alpha=0.3, discriminant > 0
    };
    for (const auto& [mu, omega2] : cases) {
        const auto exact = cpso::propagate_frozen_oscillator(1.2, -0.35, 0.4, omega2, mu, 0.8);
        const auto numerical = rk4_reference(1.2, -0.35, 0.4, omega2, mu, 0.8, 30000);
        require_near(exact.position, numerical.position, 3e-10,
                     "Closed form must agree with independent RK4 position reference");
        require_near(exact.velocity, numerical.velocity, 3e-10,
                     "Closed form must agree with independent RK4 velocity reference");

        const auto half = cpso::propagate_frozen_oscillator(1.2, -0.35, 0.4, omega2, mu, 0.4);
        const auto twice = cpso::propagate_frozen_oscillator(
            half.position, half.velocity, 0.4, omega2, mu, 0.4);
        require_near(exact.position, twice.position, 3e-12,
                     "Frozen flow must satisfy semigroup property for position");
        require_near(exact.velocity, twice.velocity, 3e-12,
                     "Frozen flow must satisfy semigroup property for velocity");
    }
}

void test_propagator_edge_cases() {
    const auto unchanged = cpso::propagate_frozen_oscillator(1.3, -0.2, 0.0, 0.4, 0.7, 0.0);
    require_near(unchanged.position, 1.3, 0.0, "dt=0 must preserve position");
    require_near(unchanged.velocity, -0.2, 0.0, "dt=0 must preserve velocity");

    const auto equilibrium = cpso::propagate_frozen_oscillator(0.4, 0.0, 0.4, 1.0, 0.5, 2.0);
    require_near(equilibrium.position, 0.4, 2e-15, "Equilibrium must remain fixed");
    require_near(equilibrium.velocity, 0.0, 2e-15, "Equilibrium velocity must remain zero");

    // Zero frequency is a valid limiting case when both random attraction terms vanish.
    const auto free_damped = cpso::propagate_frozen_oscillator(1.0, 0.5, 0.0, 0.0, 0.4, 0.7);
    const double alpha = 0.3;
    const double expected_v = 0.5 * std::exp(-2.0 * alpha * 0.7);
    const double expected_x = 1.0 + 0.5 * (1.0 - std::exp(-2.0 * alpha * 0.7)) / (2.0 * alpha);
    require_near(free_damped.position, expected_x, 2e-14, "Zero-frequency damped position");
    require_near(free_damped.velocity, expected_v, 2e-14, "Zero-frequency damped velocity");

    const auto free_undamped = cpso::propagate_frozen_oscillator(1.0, 0.5, 0.0, 0.0, 1.0, 0.7);
    require_near(free_undamped.position, 1.35, 2e-15, "Zero-frequency undamped position");
    require_near(free_undamped.velocity, 0.5, 2e-15, "Zero-frequency undamped velocity");
}

void test_boundary_map() {
    double x = -1.5;
    double v = -2.0;
    cpso::apply_clamped_restitution(x, v, -1.0, 1.0, 0.25);
    require_near(x, -1.0, 0.0, "Lower-bound crossing must clamp position");
    require_near(v, 0.5, 0.0, "Lower-bound outward velocity must reflect with restitution");

    x = 1.5;
    v = 2.0;
    cpso::apply_clamped_restitution(x, v, -1.0, 1.0, 0.5);
    require_near(x, 1.0, 0.0, "Upper-bound crossing must clamp position");
    require_near(v, -1.0, 0.0, "Upper-bound outward velocity must reflect with restitution");

    x = 1.5;
    v = -2.0;
    cpso::apply_clamped_restitution(x, v, -1.0, 1.0, 0.5);
    require_near(v, -2.0, 0.0, "Inward velocity must not be reflected");
}

void test_benchmarks() {
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Sphere, {1.0, -2.0}),
                 5.0, 1e-14, "Sphere known value");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Sphere, {0.0, 0.0}),
                 0.0, 0.0, "Sphere optimum");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, {0.0, 0.0}),
                 0.0, 1e-14, "Rastrigin optimum");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, {1.0, 1.0}),
                 2.0, 1e-12, "Rastrigin known value");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Rosenbrock, {1.0, 1.0, 1.0}),
                 0.0, 0.0, "Rosenbrock optimum");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Rosenbrock, {0.0, 0.0}),
                 1.0, 1e-14, "Rosenbrock known value");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Ackley, {0.0, 0.0, 0.0}),
                 0.0, 2e-14, "Ackley optimum");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Griewank, {0.0, 0.0}),
                 0.0, 0.0, "Griewank optimum");
    require_near(cpso::evaluate_benchmark(cpso::Benchmark::Salomon, {0.0, 0.0}),
                 0.0, 0.0, "Salomon optimum");

    bool threw = false;
    try {
        (void)cpso::evaluate_benchmark(cpso::Benchmark::Rosenbrock, {0.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Rosenbrock must reject dimension < 2");
}

void test_synthetic_work_objective() {
    const std::vector<double> x{0.25, -0.75, 1.5, 2.0};
    const double base = cpso::evaluate_benchmark(cpso::Benchmark::Sphere, x);
    require_near(cpso::evaluate_synthetic_sphere_work(x, 0U), base, 0.0,
                 "Zero-work synthetic objective must equal Sphere exactly");
    const double first = cpso::evaluate_synthetic_sphere_work(x, 10000U);
    const double second = cpso::evaluate_synthetic_sphere_work(x, 10000U);
    require(std::isfinite(first), "Synthetic work objective must remain finite");
    require(first == second, "Synthetic work objective must be deterministic");
    require(first >= base, "Synthetic objective correction must be non-negative");

    bool threw = false;
    try {
        (void)cpso::evaluate_synthetic_sphere_work(x, 100000001ULL);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Synthetic objective must reject excessive work requests");

    cpso::Config config = test_config();
    const auto objective = [](const std::vector<double>& point) {
        return cpso::evaluate_synthetic_sphere_work(point, 2000U);
    };
    const auto serial = cpso::minimize_serial(config, objective);
    for (const std::size_t workers : {1U, 2U, 4U}) {
        const auto parallel = cpso::minimize_master_slave(config, objective, workers);
        require(parallel.best_value == serial.best_value,
                "Synthetic-work master--slave best value must match serial");
        require(parallel.best_position == serial.best_position,
                "Synthetic-work master--slave position must match serial");
        require(parallel.best_history == serial.best_history,
                "Synthetic-work master--slave history must match serial");
    }
}

void test_optimizer_contract() {
    const auto config = test_config();
    const auto objective = [](const std::vector<double>& x) {
        return cpso::evaluate_benchmark(cpso::Benchmark::Sphere, x);
    };
    const auto first = cpso::minimize_serial(config, objective);
    const auto second = cpso::minimize_serial(config, objective);

    require(first.best_history == second.best_history, "Fixed-seed history must be repeatable");
    require(first.best_position == second.best_position, "Fixed-seed best position must be repeatable");
    require(first.best_value == second.best_value, "Fixed-seed best value must be repeatable");
    require(first.function_evaluations == config.population_size * (config.iterations + 1),
            "FE count must include initialization and every particle evaluation");
    require(first.best_history.size() == config.iterations + 1,
            "History must include initial state and all completed generations");
    require(first.iterations_completed == config.iterations,
            "All requested iterations must complete when early stopping is disabled");
    for (std::size_t k = 1; k < first.best_history.size(); ++k) {
        require(first.best_history[k] <= first.best_history[k - 1],
                "Elitist global best history must be non-increasing");
    }
    require(first.best_value == first.best_history.back(), "Final result must match final history entry");
    require(first.best_position.size() == config.lower_bounds.size(), "Best position dimension");
    for (std::size_t j = 0; j < first.best_position.size(); ++j) {
        require(first.best_position[j] >= config.lower_bounds[j]
                && first.best_position[j] <= config.upper_bounds[j],
                "Best position must lie within configured bounds");
    }

    std::uint64_t actual_objective_calls = 0;
    const auto counted = cpso::minimize_serial(config, [&actual_objective_calls](const std::vector<double>& x) {
        ++actual_objective_calls;
        return cpso::evaluate_benchmark(cpso::Benchmark::Sphere, x);
    });
    require(actual_objective_calls == counted.function_evaluations,
            "Reported FE count must equal the number of objective calls");
}


void test_master_slave_equivalence() {
    const auto config = test_config();
    const auto objective = [](const std::vector<double>& x) {
        return cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, x);
    };
    const auto serial = cpso::minimize_serial(config, objective);

    for (const std::size_t workers : {1U, 2U, 4U, 8U}) {
        const auto parallel = cpso::minimize_master_slave(config, objective, workers);
        require(parallel.best_value == serial.best_value,
                "Master--slave best value must exactly match the serial reference");
        require(parallel.best_position == serial.best_position,
                "Master--slave best position must exactly match the serial reference");
        require(parallel.best_history == serial.best_history,
                "Master--slave history must exactly match the serial reference");
        require(parallel.function_evaluations == serial.function_evaluations,
                "Master--slave FE count must match serial reference");
        require(parallel.iterations_completed == config.iterations,
                "Master--slave must complete all requested transitions");
        require(parallel.worker_count == workers,
                "Master--slave result must record the requested worker count");
        require(parallel.evaluation_wall_seconds >= 0.0
                    && parallel.propagation_seconds >= 0.0,
                "Master--slave timing fields must be non-negative");
    }
}

void test_master_slave_parallel_initial_evaluations() {
    cpso::Config config = test_config();
    config.population_size = 8;
    config.iterations = 0;

    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};
    const auto delayed_objective = [&active, &maximum_active](const std::vector<double>& x) {
        const int now_active = active.fetch_add(1, std::memory_order_relaxed) + 1;
        int observed = maximum_active.load(std::memory_order_relaxed);
        while (observed < now_active
               && !maximum_active.compare_exchange_weak(
                   observed, now_active, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
        active.fetch_sub(1, std::memory_order_relaxed);
        return cpso::evaluate_benchmark(cpso::Benchmark::Sphere, x);
    };

    const auto result = cpso::minimize_master_slave(config, delayed_objective, 4U);
    require(result.function_evaluations == config.population_size,
            "Zero-transition master--slave run must evaluate the initial population");
    require(maximum_active.load(std::memory_order_relaxed) >= 2,
            "Master--slave initialization must execute objective calls concurrently");
}

void test_master_slave_exception_and_validation() {
    const auto config = test_config();
    bool threw = false;
    try {
        (void)cpso::minimize_master_slave(config,
            [](const std::vector<double>&) -> double {
                throw std::runtime_error("objective failure sentinel");
            }, 3U);
    } catch (const std::runtime_error& exception) {
        threw = std::string(exception.what()) == "objective failure sentinel";
    }
    require(threw, "Master--slave must propagate objective exceptions to the caller");

    threw = false;
    try {
        (void)cpso::minimize_master_slave(config,
            [](const std::vector<double>&) {
                return std::numeric_limits<double>::quiet_NaN();
            }, 2U);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "Master--slave must reject non-finite objective values");

    threw = false;
    try {
        (void)cpso::minimize_master_slave(config,
            [](const std::vector<double>& x) { return x[0] * x[0]; }, 0U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Master--slave must reject worker_count=0");
}

void test_zero_iteration_and_single_particle() {
    cpso::Config config;
    config.lower_bounds = {-2.0};
    config.upper_bounds = {2.0};
    config.population_size = 1;
    config.iterations = 0;
    config.seed = 33;
    const auto result = cpso::minimize_serial(config, [](const std::vector<double>& x) {
        return x[0] * x[0];
    });
    require(result.iterations_completed == 0, "Zero-iteration run must complete no transitions");
    require(result.function_evaluations == 1, "Zero-iteration run evaluates only initialization");
    require(result.best_history.size() == 1, "Zero-iteration history contains only initial incumbent");
}

void test_validation_and_nonfinite_objective() {
    cpso::Config invalid = test_config();
    invalid.upper_bounds.pop_back();
    bool threw = false;
    try {
        cpso::validate_config(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Mismatched bound dimensions must be rejected");

    invalid = test_config();
    invalid.cognitive = std::numeric_limits<double>::max();
    invalid.social = std::numeric_limits<double>::max();
    threw = false;
    try {
        cpso::validate_config(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Overflowing attraction coefficient sum must be rejected");

    cpso::Config config = test_config();
    threw = false;
    try {
        (void)cpso::minimize_serial(config, [](const std::vector<double>&) {
            return std::numeric_limits<double>::quiet_NaN();
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "Non-finite objective values must be rejected");
}

}  // namespace

int main() {
    try {
        test_propagator_against_rk4();
        test_propagator_edge_cases();
        test_boundary_map();
        test_benchmarks();
        test_synthetic_work_objective();
        test_optimizer_contract();
        test_master_slave_equivalence();
        test_master_slave_parallel_initial_evaluations();
        test_master_slave_exception_and_validation();
        test_zero_iteration_and_single_particle();
        test_validation_and_nonfinite_objective();
        std::cout << "All CPSO C++ tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "TEST FAILURE: " << exception.what() << '\n';
        return 1;
    }
}
