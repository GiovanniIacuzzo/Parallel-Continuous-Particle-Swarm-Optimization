#include "cpso/analytic_propagator.hpp"
#include "cpso/objectives.hpp"
#include "cpso/optimizer.hpp"
#include "cpso/random_stream.hpp"
#include "cpso/topology.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <set>
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

void test_particle_random_stream_derivation() {
    std::set<std::pair<std::uint64_t, std::uint64_t>> fingerprints;
    for (std::uint64_t seed = 700U; seed < 704U; ++seed) {
        for (std::uint64_t particle_id = 1U; particle_id <= 40U; ++particle_id) {
            auto first = cpso::make_particle_random_engine(seed, particle_id);
            auto repeated = cpso::make_particle_random_engine(seed, particle_id);
            const std::uint64_t first_a = first();
            const std::uint64_t second_a = first();
            const std::uint64_t first_b = repeated();
            const std::uint64_t second_b = repeated();
            require(first_a == first_b && second_a == second_b,
                    "A particle stream must be repeatable for a fixed run seed and particle ID");
            require(fingerprints.emplace(first_a, second_a).second,
                    "Distinct tested run-seed/particle-ID pairs must not share the stream fingerprint");
        }
    }
    bool threw = false;
    try {
        (void)cpso::make_particle_random_engine(1U, 0U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Particle stream IDs must be one-based");
}


void require_undirected_connected(const cpso::ParticleTopology& topology, const std::string& label) {
    const std::size_t n = topology.population_size();
    std::vector<bool> visited(n, false);
    std::vector<std::size_t> queue{0U};
    visited[0] = true;
    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const std::size_t node = queue[cursor];
        const auto& neighbors = topology.neighbors(node);
        require(std::is_sorted(neighbors.begin(), neighbors.end()), label + ": neighborhoods must be sorted");
        require(std::find(neighbors.begin(), neighbors.end(), node) != neighbors.end(),
                label + ": every neighborhood must contain its own particle");
        require(std::adjacent_find(neighbors.begin(), neighbors.end()) == neighbors.end(),
                label + ": neighborhoods must not contain duplicate indices");
        for (const std::size_t neighbor : neighbors) {
            if (neighbor == node) {
                continue;
            }
            const auto& reverse = topology.neighbors(neighbor);
            require(std::find(reverse.begin(), reverse.end(), node) != reverse.end(),
                    label + ": graph must be undirected");
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    require(queue.size() == n, label + ": graph must be connected");
}

void test_extended_topology_families() {
    const cpso::ParticleTopology wheel(cpso::TopologyKind::Wheel, 8U);
    require_undirected_connected(wheel, "wheel");
    std::vector<std::size_t> wheel_degrees;
    for (std::size_t i = 0; i < 8U; ++i) {
        wheel_degrees.push_back(wheel.neighbors(i).size() - 1U);
    }
    std::sort(wheel_degrees.begin(), wheel_degrees.end());
    require(wheel_degrees.back() == 7U,
            "PSO wheel must contain one focal particle adjacent to every peripheral particle");
    require(std::count(wheel_degrees.begin(), wheel_degrees.end(), 1U) == 7,
            "PSO wheel peripheral nodes must connect only to the focal particle");

    const cpso::ParticleTopology random_a(cpso::TopologyKind::RandomConnected, 40U, 0U, 0U, 12345U, 4U);
    const cpso::ParticleTopology random_b(cpso::TopologyKind::RandomConnected, 40U, 0U, 0U, 12345U, 4U);
    require_undirected_connected(random_a, "random-connected");
    for (std::size_t i = 0; i < 40U; ++i) {
        require(random_a.neighbors(i) == random_b.neighbors(i),
                "Random topology must be deterministic for a fixed topology seed");
    }

    const cpso::ParticleTopology small_world_ring(
        cpso::TopologyKind::SmallWorld, 12U, 0U, 0U, 77U, 4U, 0.0);
    require_undirected_connected(small_world_ring, "small-world beta=0");
    require(small_world_ring.neighbors(0U) == std::vector<std::size_t>({0U, 1U, 2U, 10U, 11U}),
            "Watts-Strogatz beta=0 must recover a degree-four periodic ring lattice");
    const cpso::ParticleTopology small_world(
        cpso::TopologyKind::SmallWorld, 40U, 0U, 0U, 991U, 4U, 0.2);
    std::size_t small_world_degree_sum = 0U;
    for (std::size_t i = 0; i < 40U; ++i) {
        require(std::find(small_world.neighbors(i).begin(), small_world.neighbors(i).end(), i)
                    != small_world.neighbors(i).end(),
                "Small-world neighborhoods must include self");
        small_world_degree_sum += small_world.neighbors(i).size() - 1U;
    }
    require(small_world_degree_sum == 40U * 4U,
            "Watts-Strogatz rewiring must preserve the total number of undirected edges");

    const cpso::ParticleTopology scale_free(cpso::TopologyKind::ScaleFree, 40U, 0U, 0U, 2026U, 4U);
    const cpso::ParticleTopology scale_free_replay(cpso::TopologyKind::ScaleFree, 40U, 0U, 0U, 2026U, 4U);
    require_undirected_connected(scale_free, "scale-free");
    std::vector<std::size_t> scale_free_degrees;
    for (std::size_t i = 0; i < 40U; ++i) {
        scale_free_degrees.push_back(scale_free.neighbors(i).size() - 1U);
        require(scale_free.neighbors(i) == scale_free_replay.neighbors(i),
                "Scale-free topology must be deterministic for a fixed seed");
    }
    const auto [minimum_degree, maximum_degree] = std::minmax_element(
        scale_free_degrees.begin(), scale_free_degrees.end());
    require(*maximum_degree > *minimum_degree,
            "Scale-free topology should have a non-uniform degree distribution");
}

void test_topology_neighborhoods() {
    const cpso::ParticleTopology global(cpso::TopologyKind::GlobalBest, 4U);
    const auto* shared_global_neighbors = &global.neighbors(0U);
    for (std::size_t i = 0; i < 4U; ++i) {
        require(global.neighbors(i) == std::vector<std::size_t>({0U, 1U, 2U, 3U}),
                "Global topology must connect every particle to the full population");
        require(&global.neighbors(i) == shared_global_neighbors,
                "Global topology must reuse one shared O(N) index list");
    }

    const cpso::ParticleTopology ring(cpso::TopologyKind::Ring, 4U);
    require(ring.neighbors(0U) == std::vector<std::size_t>({0U, 1U, 3U}),
            "Ring topology must wrap around at the left boundary");
    require(ring.neighbors(2U) == std::vector<std::size_t>({1U, 2U, 3U}),
            "Ring topology must include self and immediate neighbors");

    const cpso::ParticleTopology ring_two(cpso::TopologyKind::Ring, 2U);
    require(ring_two.neighbors(0U) == std::vector<std::size_t>({0U, 1U}),
            "Two-particle ring must deduplicate periodic neighbors");
    const cpso::ParticleTopology ring_one(cpso::TopologyKind::Ring, 1U);
    require(ring_one.neighbors(0U) == std::vector<std::size_t>({0U}),
            "One-particle ring must contain only itself");

    const cpso::ParticleTopology grid(cpso::TopologyKind::GridVonNeumann, 9U, 3U, 3U);
    require(grid.neighbors(0U) == std::vector<std::size_t>({0U, 1U, 2U, 3U, 6U}),
            "Toroidal 3x3 von-Neumann neighborhood at the upper-left corner");
    require(grid.neighbors(4U) == std::vector<std::size_t>({1U, 3U, 4U, 5U, 7U}),
            "Toroidal 3x3 von-Neumann neighborhood at the center");
    const cpso::ParticleTopology grid_two(cpso::TopologyKind::GridVonNeumann, 4U, 2U, 2U);
    require(grid_two.neighbors(0U) == std::vector<std::size_t>({0U, 1U, 2U}),
            "2x2 toroidal grid must deduplicate coincident neighbors");

    const auto leaders = ring.select_local_leaders({3.0, 1.0, 2.0, 0.5});
    require(leaders == std::vector<std::size_t>({3U, 1U, 3U, 3U}),
            "Ring social leaders must be the lowest-valued personal best in each neighborhood");
    const auto tied_leaders = ring.select_local_leaders({1.0, 1.0, 1.0, 1.0});
    require(tied_leaders == std::vector<std::size_t>({0U, 0U, 1U, 0U}),
            "Local leader ties must be resolved by smallest particle index");
}

void test_topology_validation() {
    bool threw = false;
    try {
        (void)cpso::ParticleTopology(cpso::TopologyKind::GridVonNeumann, 10U, 3U, 3U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Grid shape must multiply to population size");

    threw = false;
    try {
        (void)cpso::ParticleTopology(cpso::TopologyKind::GridVonNeumann, 9U, 0U, 9U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Grid dimensions must be positive");

    threw = false;
    try {
        (void)cpso::ParticleTopology(cpso::TopologyKind::GridVonNeumann, 10U, 1U, 10U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Grid topology must reject a degenerate one-dimensional shape");

    threw = false;
    try {
        (void)cpso::ParticleTopology(cpso::TopologyKind::Ring, 9U, 3U, 3U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Grid shape must not be silently accepted by ring topology");

    const cpso::ParticleTopology ring(cpso::TopologyKind::Ring, 4U);
    threw = false;
    try {
        (void)ring.select_local_leaders({1.0, 2.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Leader selection must reject a population-size mismatch");

    threw = false;
    try {
        (void)ring.select_local_leaders({1.0, std::numeric_limits<double>::infinity(), 2.0, 3.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Leader selection must reject non-finite personal-best values");
}

void test_topology_optimizer_and_parallel_equivalence() {
    cpso::Config config = test_config();
    const auto objective = [](const std::vector<double>& x) {
        return cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, x);
    };

    config.topology = cpso::TopologyKind::GlobalBest;
    const auto global = cpso::minimize_serial(config, objective);
    for (const std::size_t workers : {1U, 2U, 4U}) {
        const auto parallel = cpso::minimize_master_slave(config, objective, workers);
        require(parallel.best_value == global.best_value,
                "Global topology master-slave value must match serial");
        require(parallel.best_position == global.best_position,
                "Global topology master-slave position must match serial");
        require(parallel.best_history == global.best_history,
                "Global topology master-slave history must match serial");
    }

    config.topology = cpso::TopologyKind::Ring;
    const auto ring = cpso::minimize_serial(config, objective);
    for (const std::size_t workers : {1U, 2U, 4U}) {
        const auto parallel = cpso::minimize_master_slave(config, objective, workers);
        require(parallel.best_value == ring.best_value,
                "Ring topology master-slave value must match serial");
        require(parallel.best_position == ring.best_position,
                "Ring topology master-slave position must match serial");
        require(parallel.best_history == ring.best_history,
                "Ring topology master-slave history must match serial");
    }

    config.topology = cpso::TopologyKind::GridVonNeumann;
    config.grid_rows = 4U;
    config.grid_columns = 8U;
    const auto grid = cpso::minimize_serial(config, objective);
    for (const std::size_t workers : {1U, 2U, 4U}) {
        const auto parallel = cpso::minimize_master_slave(config, objective, workers);
        require(parallel.best_value == grid.best_value,
                "Grid topology master-slave value must match serial");
        require(parallel.best_position == grid.best_position,
                "Grid topology master-slave position must match serial");
        require(parallel.best_history == grid.best_history,
                "Grid topology master-slave history must match serial");
    }

    require(global.best_history != ring.best_history || global.best_position != ring.best_position,
            "Global and ring social topologies should produce distinguishable trajectories on this test case");
    require(global.best_history != grid.best_history || global.best_position != grid.best_position,
            "Global and grid social topologies should produce distinguishable trajectories on this test case");
    require(ring.best_history != grid.best_history || ring.best_position != grid.best_position,
            "Ring and grid social topologies should produce distinguishable trajectories on this test case");
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
    require(first.elapsed_seconds >= 0.0
                && first.evaluation_wall_seconds >= 0.0
                && first.propagation_seconds >= 0.0
                && first.topology_setup_seconds >= 0.0
                && first.social_selection_seconds >= 0.0,
            "All recorded timing fields must be non-negative");
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
                    && parallel.propagation_seconds >= 0.0
                    && parallel.topology_setup_seconds >= 0.0
                    && parallel.social_selection_seconds >= 0.0,
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


void test_particle_parallel_exact_equivalence() {
    const cpso::Config base = test_config();
    const std::vector<cpso::TopologyKind> topologies = {
        cpso::TopologyKind::GlobalBest,
        cpso::TopologyKind::Ring,
        cpso::TopologyKind::GridVonNeumann,
        cpso::TopologyKind::Wheel,
        cpso::TopologyKind::RandomConnected,
        cpso::TopologyKind::SmallWorld,
        cpso::TopologyKind::ScaleFree,
    };
    for (const cpso::TopologyKind topology : topologies) {
        cpso::Config config = base;
        config.topology = topology;
        if (topology == cpso::TopologyKind::GridVonNeumann) {
            config.population_size = 20U;
            config.grid_rows = 4U;
            config.grid_columns = 5U;
        }
        const auto objective = [](const std::vector<double>& x) {
            return cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, x);
        };
        const cpso::Result serial = cpso::minimize_serial(config, objective);
        for (const std::size_t workers : {1U, 2U, 4U, 8U}) {
            const cpso::Result parallel = cpso::minimize_particle_parallel(
                config, objective, workers);
            require(parallel.best_value == serial.best_value,
                    "Particle-parallel best value must match serial exactly");
            require(parallel.best_position == serial.best_position,
                    "Particle-parallel position must match serial exactly");
            require(parallel.best_history == serial.best_history,
                    "Particle-parallel history must match serial exactly");
            require(parallel.history_evaluations == serial.history_evaluations,
                    "Particle-parallel history FE axis must match serial exactly");
            require(parallel.function_evaluations == serial.function_evaluations,
                    "Particle-parallel FE count must match serial");
            require(parallel.worker_count == workers,
                    "Particle-parallel worker metadata must be correct");
        }
    }
}

void test_asynchronous_contract_and_repeatability() {
    cpso::Config config = test_config();
    config.topology = cpso::TopologyKind::Ring;
    const auto objective = [](const std::vector<double>& x) {
        return cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, x);
    };
    const cpso::Result first = cpso::minimize_asynchronous(config, objective, 1U);
    const cpso::Result second = cpso::minimize_asynchronous(config, objective, 1U);
    require(first.best_value == second.best_value,
            "Single-worker asynchronous mode must be repeatable");
    require(first.best_position == second.best_position,
            "Single-worker asynchronous position must be repeatable");
    require(first.best_history == second.best_history,
            "Single-worker asynchronous event history must be repeatable");

    const cpso::Result parallel = cpso::minimize_asynchronous(config, objective, 4U);
    const std::uint64_t expected_fe = static_cast<std::uint64_t>(config.population_size)
        * static_cast<std::uint64_t>(config.iterations + 1U);
    require(parallel.function_evaluations == expected_fe,
            "Asynchronous FE count must equal population times updates plus initialization");
    require(parallel.best_history.size() == 1U
                + config.population_size * config.iterations,
            "Asynchronous history must contain initial best and every committed update");
    require(parallel.history_evaluations.size() == parallel.best_history.size(),
            "Asynchronous history and FE axis lengths must match");
    require(parallel.history_evaluations.front() == config.population_size,
            "Asynchronous history must begin after initial population evaluation");
    require(parallel.history_evaluations.back() == expected_fe,
            "Asynchronous history must end at total FE count");
    for (std::size_t i = 1; i < parallel.history_evaluations.size(); ++i) {
        require(parallel.history_evaluations[i] == parallel.history_evaluations[i - 1U] + 1U,
                "Asynchronous event FE axis must increment one evaluation at a time");
        require(parallel.best_history[i] <= parallel.best_history[i - 1U],
                "Asynchronous best-so-far history must be non-increasing");
    }
    require(parallel.best_value == parallel.best_history.back(),
            "Asynchronous final best must equal the last event-history value");
    require(parallel.iterations_completed == config.iterations,
            "Asynchronous completed iterations are per-particle update counts");
}

void test_island_model_contract_and_migration() {
    cpso::Config config = test_config();
    config.population_size = 24U;
    config.iterations = 10U;
    const auto objective = [](const std::vector<double>& x) {
        return cpso::evaluate_benchmark(cpso::Benchmark::Sphere, x);
    };

    cpso::IslandConfig no_migration;
    no_migration.island_count = 4U;
    no_migration.worker_count = 4U;
    no_migration.migration_interval = 0U;
    const cpso::Result independent = cpso::minimize_islands(config, objective, no_migration);
    require(independent.function_evaluations == 24U * 11U,
            "Island model without migration must preserve the total FE budget");
    require(independent.best_history.size() == config.iterations + 1U,
            "Island history must be generation-synchronous");
    require(independent.history_evaluations.back() == independent.function_evaluations,
            "Island history FE axis must terminate at total FE count");
    require(independent.migrations_performed == 0U,
            "Disabled migration must not perform any migration");

    cpso::IslandConfig ring_migration;
    ring_migration.island_count = 4U;
    ring_migration.worker_count = 4U;
    ring_migration.migration_interval = 2U;
    ring_migration.migration_topology = cpso::MigrationTopology::Ring;
    const cpso::Result migrated = cpso::minimize_islands(config, objective, ring_migration);
    require(migrated.function_evaluations == independent.function_evaluations,
            "Migration must not consume objective evaluations");
    require(migrated.best_value <= independent.best_value + 1e-14,
            "Elitist best-to-worst migration cannot worsen the global incumbent");
    require(migrated.migrations_performed <= 4U * 4U,
            "Migration counter must not exceed destinations times migration events");

    cpso::IslandConfig fully_connected = ring_migration;
    fully_connected.migration_topology = cpso::MigrationTopology::FullyConnected;
    const cpso::Result all_to_all = cpso::minimize_islands(config, objective, fully_connected);
    require(all_to_all.function_evaluations == independent.function_evaluations,
            "Fully-connected migration must preserve FE budget");
    require(all_to_all.migrations_performed <= 4U * 4U,
            "Fully-connected migration counter must respect migration-event bound");
    const cpso::Result all_to_all_repeat = cpso::minimize_islands(config, objective, fully_connected);
    require(all_to_all.best_value == all_to_all_repeat.best_value,
            "Synchronous island model must be deterministic for a fixed seed/configuration");
    require(all_to_all.best_position == all_to_all_repeat.best_position,
            "Synchronous island best position must be reproducible");
    require(all_to_all.migrations_performed == all_to_all_repeat.migrations_performed,
            "Synchronous island migration count must be reproducible");
}

void test_island_validation() {
    cpso::Config config = test_config();
    cpso::IslandConfig islands;
    islands.island_count = 4U;
    islands.worker_count = 4U;
    bool threw = false;
    try {
        config.population_size = 19U;
        (void)cpso::minimize_islands(config,
            [](const std::vector<double>& x) { return x[0] * x[0]; }, islands);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Island population must divide evenly across islands");

    config = test_config();
    config.topology = cpso::TopologyKind::GridVonNeumann;
    config.population_size = 20U;
    config.grid_rows = 4U;
    config.grid_columns = 5U;
    threw = false;
    try {
        (void)cpso::minimize_islands(config,
            [](const std::vector<double>& x) { return x[0] * x[0]; }, islands);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Grid particle topology must be rejected inside the current island backend");
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
        test_particle_random_stream_derivation();
        test_topology_neighborhoods();
        test_extended_topology_families();
        test_topology_validation();
        test_topology_optimizer_and_parallel_equivalence();
        test_master_slave_parallel_initial_evaluations();
        test_master_slave_exception_and_validation();
        test_zero_iteration_and_single_particle();
        test_validation_and_nonfinite_objective();
        test_particle_parallel_exact_equivalence();
        test_asynchronous_contract_and_repeatability();
        test_island_model_contract_and_migration();
        test_island_validation();
        std::cout << "All CPSO C++ tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "TEST FAILURE: " << exception.what() << '\n';
        return 1;
    }
}
