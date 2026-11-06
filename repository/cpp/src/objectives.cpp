#include "cpso/objectives.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cpso {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

void validate_vector(const std::vector<double>& x) {
    if (x.empty()) {
        throw std::invalid_argument("Benchmark input vector must not be empty");
    }
    for (const double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Benchmark input contains a non-finite value");
        }
    }
}
}  // namespace

double evaluate_benchmark(const Benchmark benchmark, const std::vector<double>& x) {
    validate_vector(x);
    const std::size_t dimension = x.size();

    switch (benchmark) {
        case Benchmark::Sphere: {
            double sum = 0.0;
            for (const double value : x) {
                sum += value * value;
            }
            return sum;
        }
        case Benchmark::Rastrigin: {
            double sum = 10.0 * static_cast<double>(dimension);
            for (const double value : x) {
                sum += value * value - 10.0 * std::cos(2.0 * kPi * value);
            }
            return sum;
        }
        case Benchmark::Rosenbrock: {
            if (dimension < 2) {
                throw std::invalid_argument("Rosenbrock benchmark requires dimension >= 2");
            }
            double sum = 0.0;
            for (std::size_t i = 0; i + 1 < dimension; ++i) {
                const double a = x[i + 1] - x[i] * x[i];
                const double b = 1.0 - x[i];
                sum += 100.0 * a * a + b * b;
            }
            return sum;
        }
        case Benchmark::Ackley: {
            double sum_squares = 0.0;
            double sum_cosines = 0.0;
            for (const double value : x) {
                sum_squares += value * value;
                sum_cosines += std::cos(2.0 * kPi * value);
            }
            const double d = static_cast<double>(dimension);
            return -20.0 * std::exp(-0.2 * std::sqrt(sum_squares / d))
                   - std::exp(sum_cosines / d) + 20.0 + std::exp(1.0);
        }
        case Benchmark::Griewank: {
            double sum = 0.0;
            double product = 1.0;
            for (std::size_t i = 0; i < dimension; ++i) {
                const double value = x[i];
                sum += value * value / 4000.0;
                product *= std::cos(value / std::sqrt(static_cast<double>(i + 1)));
            }
            return sum - product + 1.0;
        }
        case Benchmark::Salomon: {
            double norm_squared = 0.0;
            for (const double value : x) {
                norm_squared += value * value;
            }
            const double norm = std::sqrt(norm_squared);
            return 1.0 - std::cos(2.0 * kPi * norm) + 0.1 * norm;
        }
    }
    throw std::invalid_argument("Unknown benchmark identifier");
}


double evaluate_synthetic_sphere_work(
    const std::vector<double>& x, const std::uint64_t work_iterations) {
    validate_vector(x);
    constexpr std::uint64_t kMaximumWorkIterations = 100'000'000ULL;
    if (work_iterations > kMaximumWorkIterations) {
        throw std::invalid_argument("synthetic objective work_iterations exceeds 100000000");
    }

    double sphere_value = 0.0;
    for (const double value : x) {
        sphere_value += value * value;
    }
    if (work_iterations == 0U) {
        return sphere_value;
    }

    // A loop-carried floating-point recurrence provides a tunable arithmetic
    // workload. The tiny positive correction makes the recurrence observable
    // in the returned value, preventing dead-code elimination without
    // materially changing the Sphere landscape at ordinary optimization scales.
    // This synthetic function must not be reported as a standard Sphere result.
    double accumulator = 0.125;
    std::size_t coordinate = 0U;
    for (std::uint64_t i = 0U; i < work_iterations; ++i) {
        const double drive = std::fma(
            x[coordinate], 1.0e-9,
            static_cast<double>((i & 255U) + 1U) * 1.0e-12);
        accumulator = std::fma(accumulator, 0.999999, drive);
        ++coordinate;
        if (coordinate == x.size()) {
            coordinate = 0U;
        }
    }
    return sphere_value + 1.0e-30 * std::abs(accumulator);
}

Benchmark benchmark_from_name(const std::string_view name) {
    if (name == "sphere") return Benchmark::Sphere;
    if (name == "rastrigin") return Benchmark::Rastrigin;
    if (name == "rosenbrock") return Benchmark::Rosenbrock;
    if (name == "ackley") return Benchmark::Ackley;
    if (name == "griewank") return Benchmark::Griewank;
    if (name == "salomon") return Benchmark::Salomon;
    throw std::invalid_argument("Unknown benchmark name: " + std::string(name));
}

std::string_view benchmark_name(const Benchmark benchmark) {
    switch (benchmark) {
        case Benchmark::Sphere: return "sphere";
        case Benchmark::Rastrigin: return "rastrigin";
        case Benchmark::Rosenbrock: return "rosenbrock";
        case Benchmark::Ackley: return "ackley";
        case Benchmark::Griewank: return "griewank";
        case Benchmark::Salomon: return "salomon";
    }
    throw std::invalid_argument("Unknown benchmark identifier");
}

}  // namespace cpso
