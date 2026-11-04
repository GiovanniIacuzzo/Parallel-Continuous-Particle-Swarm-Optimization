#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace cpso {

enum class Benchmark : int {
    Sphere = 1,
    Rastrigin = 2,
    Rosenbrock = 3,
    Ackley = 4,
    Griewank = 5,
    Salomon = 6
};

[[nodiscard]] double evaluate_benchmark(Benchmark benchmark, const std::vector<double>& x);
// Controlled synthetic objective for execution-cost characterization. It is
// Sphere plus a negligible deterministic term that keeps the arithmetic loop
// observable to optimizing compilers. This is a performance instrument, not
// a standard benchmark function.
[[nodiscard]] double evaluate_synthetic_sphere_work(
    const std::vector<double>& x, std::uint64_t work_iterations);
[[nodiscard]] Benchmark benchmark_from_name(std::string_view name);
[[nodiscard]] std::string_view benchmark_name(Benchmark benchmark);

}  // namespace cpso
