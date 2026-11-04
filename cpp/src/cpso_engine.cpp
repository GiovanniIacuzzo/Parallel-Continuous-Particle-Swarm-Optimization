#include "cpso_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <chrono>

#ifdef CPSO_WITH_OPENMP
  #include <omp.h>
#endif

namespace cpso {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSchwefelStar = 420.9687462275036;

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

struct Particle {
    std::vector<double> x;
    std::vector<double> velocity;
    std::vector<double> pbest;
    double value{std::numeric_limits<double>::infinity()};
    double pbest_value{std::numeric_limits<double>::infinity()};
    std::mt19937_64 rng;
};

struct Leader {
    std::vector<double> position;
    double value{std::numeric_limits<double>::infinity()};
};

inline double uniform01(std::mt19937_64 &rng) {
    // 53-bit generation on [0,1), stable and independent of OpenMP scheduling.
    return std::generate_canonical<double, 53>(rng);
}

void validate(const CpsoConfig &c) {
    if (c.abi_version != 1U) throw std::invalid_argument("Unsupported C API version.");
    if (c.dimension < 1) throw std::invalid_argument("dimension must be >= 1.");
    if (c.population_size < 1) throw std::invalid_argument("population_size must be >= 1.");
    if (c.workers < 1) throw std::invalid_argument("workers must be >= 1.");
    if (c.population_size < c.workers)
        throw std::invalid_argument("population_size must be >= workers so every island owns at least one particle.");
    if (c.iterations < 0) throw std::invalid_argument("iterations must be >= 0.");
    if (c.sync_interval < 1) throw std::invalid_argument("sync_interval must be >= 1.");
    if (c.topology < CPSO_TOPOLOGY_INDEPENDENT || c.topology > CPSO_TOPOLOGY_RING)
        throw std::invalid_argument("Unknown topology.");
    if (c.benchmark < CPSO_BENCHMARK_SPHERE || c.benchmark > CPSO_BENCHMARK_SALOMON)
        throw std::invalid_argument("Unknown benchmark.");
    if (c.lower_bounds == nullptr || c.upper_bounds == nullptr || c.optimum_position == nullptr)
        throw std::invalid_argument("Bounds and optimum_position pointers must not be null.");
    if (!(c.mu_max >= c.mu_min) || !std::isfinite(c.mu_max) || !std::isfinite(c.mu_min)
        || c.mu_min < 0.0 || c.mu_max > 1.0)
        throw std::invalid_argument("For a non-negative damping coefficient require 0 <= mu_min <= mu_max <= 1.");
    if (c.cognitive < 0.0 || c.social < 0.0 || !std::isfinite(c.cognitive) || !std::isfinite(c.social)
        || (c.cognitive + c.social) <= 0.0)
        throw std::invalid_argument("Cognitive/social coefficients must be finite, non-negative and not both zero.");
    if (!(c.dt > 0.0) || !std::isfinite(c.dt)) throw std::invalid_argument("dt must be finite and > 0.");
    if (c.initial_velocity_fraction < 0.0 || !std::isfinite(c.initial_velocity_fraction))
        throw std::invalid_argument("initial_velocity_fraction must be finite and >= 0.");
    if (c.boundary_restitution < 0.0 || c.boundary_restitution > 1.0 || !std::isfinite(c.boundary_restitution))
        throw std::invalid_argument("boundary_restitution must lie in [0,1].");
    if (c.benchmark == CPSO_BENCHMARK_ROSENBROCK && c.dimension < 2)
        throw std::invalid_argument("Rosenbrock requires dimension >= 2.");
    for (int d = 0; d < c.dimension; ++d) {
        const double lo = c.lower_bounds[d];
        const double hi = c.upper_bounds[d];
        const double opt = c.optimum_position[d];
        if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
            throw std::invalid_argument("Every bound must be finite and lower < upper.");
        if (!std::isfinite(opt) || opt < lo || opt > hi)
            throw std::invalid_argument("Every optimum coordinate must be finite and inside the bounds.");
    }
}

// Exact flow over one interval for the frozen-coefficient oscillator:
// u'' + 2*alpha*u' + omega^2*u = 0, u=x-x_eq, alpha=(1-mu)/2.
// This is exact for the local frozen system; it is not an exact solution of
// the full switched/stochastic CPSO process across best-position updates.
void exact_oscillator_step_impl(double x, double v, double equilibrium,
                           double omega2, double mu, double dt,
                           double &x_next, double &v_next) {
    const double alpha = 0.5 * (1.0 - mu);
    const double discriminant = alpha * alpha - omega2;
    const double u0 = x - equilibrium;
    const double eps_scale = 64.0 * std::numeric_limits<double>::epsilon()
                           * std::max({1.0, alpha * alpha, omega2});

    if (std::abs(discriminant) <= eps_scale) {
        // Critically damped repeated root r=-alpha.
        const double c2 = v + alpha * u0;
        const double e = std::exp(-alpha * dt);
        const double u = e * (u0 + c2 * dt);
        const double du = e * (v - alpha * c2 * dt);
        x_next = equilibrium + u;
        v_next = du;
    } else if (discriminant < 0.0) {
        // Underdamped: use real trigonometric form (avoids complex arithmetic).
        const double beta = std::sqrt(-discriminant);
        const double c = (v + alpha * u0) / beta;
        const double cs = std::cos(beta * dt);
        const double sn = std::sin(beta * dt);
        const double e = std::exp(-alpha * dt);
        const double q = u0 * cs + c * sn;
        const double dq = -u0 * beta * sn + c * beta * cs;
        x_next = equilibrium + e * q;
        v_next = e * (dq - alpha * q);
    } else {
        // Overdamped: two distinct real roots.
        const double root = std::sqrt(discriminant);
        const double r1 = -alpha + root;
        const double r2 = -alpha - root;
        const double c1 = (v - r2 * u0) / (r1 - r2);
        const double c2 = (r1 * u0 - v) / (r1 - r2);
        const double e1 = std::exp(r1 * dt);
        const double e2 = std::exp(r2 * dt);
        x_next = equilibrium + c1 * e1 + c2 * e2;
        v_next = c1 * r1 * e1 + c2 * r2 * e2;
    }
}

void apply_clamped_restitution(double &x, double &v, double lo, double hi, double restitution) {
    if (x < lo) {
        x = lo;
        if (v < 0.0) v = -restitution * v;
    } else if (x > hi) {
        x = hi;
        if (v > 0.0) v = -restitution * v;
    }
}

Leader better_of(const Leader &a, const Leader &b) {
    return (a.value <= b.value) ? a : b;
}

} // namespace

void exact_oscillator_step(double x, double v, double equilibrium,
                           double omega2, double mu, double dt,
                           double &x_next, double &v_next) {
    exact_oscillator_step_impl(x, v, equilibrium, omega2, mu, dt, x_next, v_next);
}

double evaluate_benchmark(int benchmark, const std::vector<double> &x,
                          const std::vector<double> &optimum) {
    const std::size_t dim = x.size();
    if (dim != optimum.size()) throw std::invalid_argument("Benchmark vector dimension mismatch.");
    std::vector<double> y(dim);
    switch (benchmark) {
        case CPSO_BENCHMARK_SPHERE: {
            double sum = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i];
                sum += z * z;
            }
            return sum;
        }
        case CPSO_BENCHMARK_RASTRIGIN: {
            double sum = 10.0 * static_cast<double>(dim);
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i];
                sum += z * z - 10.0 * std::cos(2.0 * kPi * z);
            }
            return sum;
        }
        case CPSO_BENCHMARK_ROSENBROCK: {
            for (std::size_t i = 0; i < dim; ++i) y[i] = x[i] - optimum[i] + 1.0;
            double sum = 0.0;
            for (std::size_t i = 0; i + 1 < dim; ++i) {
                const double a = y[i + 1] - y[i] * y[i];
                const double b = 1.0 - y[i];
                sum += 100.0 * a * a + b * b;
            }
            return sum;
        }
        case CPSO_BENCHMARK_SCHWEFEL: {
            const double base = kSchwefelStar * std::sin(std::sqrt(kSchwefelStar));
            double sum = static_cast<double>(dim) * base;
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i] + kSchwefelStar;
                sum -= z * std::sin(std::sqrt(std::abs(z)));
            }
            return sum;
        }
        case CPSO_BENCHMARK_GRIEWANK: {
            double sum = 0.0;
            double product = 1.0;
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i];
                sum += z * z / 4000.0;
                product *= std::cos(z / std::sqrt(static_cast<double>(i + 1)));
            }
            return sum - product + 1.0;
        }
        case CPSO_BENCHMARK_ACKLEY: {
            double sum_sq = 0.0;
            double sum_cos = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i];
                sum_sq += z * z;
                sum_cos += std::cos(2.0 * kPi * z);
            }
            const double d = static_cast<double>(dim);
            return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / d))
                   - std::exp(sum_cos / d) + 20.0 + std::exp(1.0);
        }
        case CPSO_BENCHMARK_SALOMON: {
            double norm_sq = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                const double z = x[i] - optimum[i];
                norm_sq += z * z;
            }
            const double norm = std::sqrt(norm_sq);
            return 1.0 - std::cos(2.0 * kPi * norm) + 0.1 * norm;
        }
        default:
            throw std::invalid_argument("Unknown benchmark id.");
    }
}

EngineResult run(const CpsoConfig &c) {
    validate(c);
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();

    const int d = c.dimension;
    const int n = c.population_size;
    const int p = c.workers;
    const int sync_interval = c.sync_interval;
#ifdef CPSO_WITH_OPENMP
    omp_set_dynamic(0);
#endif

    std::vector<double> lower(c.lower_bounds, c.lower_bounds + d);
    std::vector<double> upper(c.upper_bounds, c.upper_bounds + d);
    std::vector<double> optimum(c.optimum_position, c.optimum_position + d);

    // Contiguous, deterministic partition of a fixed global population.
    std::vector<int> island_of_particle(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int island = std::min(p - 1, static_cast<int>((static_cast<int64_t>(i) * p) / n));
        island_of_particle[static_cast<std::size_t>(i)] = island;
    }

    std::vector<Particle> particles(static_cast<std::size_t>(n));
    std::vector<Leader> leaders(static_cast<std::size_t>(p));
    std::vector<Leader> targets(static_cast<std::size_t>(p));
    double compute_seconds = 0.0;
    double communication_seconds = 0.0;

    // Initialization uses one deterministic random stream per particle, so the
    // Star topology is invariant to the number and scheduling of OpenMP workers.
    const auto initialization_start = clock::now();
#ifdef CPSO_WITH_OPENMP
    #pragma omp parallel for schedule(static) num_threads(p)
#endif
    for (int i = 0; i < n; ++i) {
        Particle &part = particles[static_cast<std::size_t>(i)];
        part.rng.seed(splitmix64(c.seed ^ static_cast<uint64_t>(i + 1)));
        part.x.resize(static_cast<std::size_t>(d));
        part.velocity.resize(static_cast<std::size_t>(d));
        part.pbest.resize(static_cast<std::size_t>(d));
        for (int j = 0; j < d; ++j) {
            const double lo = lower[static_cast<std::size_t>(j)];
            const double hi = upper[static_cast<std::size_t>(j)];
            const double width = hi - lo;
            const double r_pos = uniform01(part.rng);
            const double r_vel = uniform01(part.rng);
            part.x[static_cast<std::size_t>(j)] = lo + r_pos * width;
            part.velocity[static_cast<std::size_t>(j)] =
                (2.0 * r_vel - 1.0) * c.initial_velocity_fraction * width;
        }
        part.value = evaluate_benchmark(c.benchmark, part.x, optimum);
        part.pbest = part.x;
        part.pbest_value = part.value;
    }

    // One O(N) deterministic reduction, rather than scanning the whole
    // population once per island (O(P*N)).
    for (int i = 0; i < n; ++i) {
        const Particle &part = particles[static_cast<std::size_t>(i)];
        const int island = island_of_particle[static_cast<std::size_t>(i)];
        Leader &leader = leaders[static_cast<std::size_t>(island)];
        if (part.pbest_value < leader.value) {
            leader.value = part.pbest_value;
            leader.position = part.pbest;
        }
    }
    compute_seconds += std::chrono::duration<double>(clock::now() - initialization_start).count();

    for (int k = 0; k < p; ++k) targets[static_cast<std::size_t>(k)] = leaders[static_cast<std::size_t>(k)];
    if (c.topology == CPSO_TOPOLOGY_STAR) {
        Leader initial_global;
        for (const Leader &leader : leaders) initial_global = better_of(initial_global, leader);
        for (int k = 0; k < p; ++k) targets[static_cast<std::size_t>(k)] = initial_global;
    }

    auto global_leader = [&]() {
        Leader best;
        for (const Leader &leader : leaders) best = better_of(best, leader);
        return best;
    };

    EngineResult out;
    out.history.reserve(static_cast<std::size_t>(c.iterations + 1));
    out.history.push_back(global_leader().value);
    out.function_evaluations = static_cast<uint64_t>(n);

    for (int iter = 0; iter < c.iterations; ++iter) {
        const auto compute_start = clock::now();
        const double progress = (c.iterations <= 1)
            ? 1.0 : static_cast<double>(iter) / static_cast<double>(c.iterations - 1);
        const double mu = c.mu_max - (c.mu_max - c.mu_min) * progress;

#ifdef CPSO_WITH_OPENMP
        #pragma omp parallel for schedule(static) num_threads(p)
#endif
        for (int i = 0; i < n; ++i) {
            Particle &part = particles[static_cast<std::size_t>(i)];
            const int island = island_of_particle[static_cast<std::size_t>(i)];
            const Leader &leader = targets[static_cast<std::size_t>(island)];
            // r1 and r2 are scalar per particle and per time interval, shared
            // by the coordinates of that particle. This matches the vector ODE
            // with scalar damping/frequency coefficients used in the source CPSO.
            const double r1 = uniform01(part.rng);
            const double r2 = uniform01(part.rng);
            const double omega2 = c.cognitive * r1 + c.social * r2;
            for (int j = 0; j < d; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                if (omega2 <= std::numeric_limits<double>::min()) {
                    // Degenerate zero acceleration: free inertial motion.
                    double x_next = part.x[jj] + part.velocity[jj] * c.dt;
                    double v_next = part.velocity[jj];
                    apply_clamped_restitution(x_next, v_next, lower[jj], upper[jj], c.boundary_restitution);
                    part.x[jj] = x_next;
                    part.velocity[jj] = v_next;
                    continue;
                }
                const double forcing = c.cognitive * r1 * part.pbest[jj]
                                     + c.social * r2 * leader.position[jj];
                const double equilibrium = forcing / omega2;
                double x_next = 0.0;
                double v_next = 0.0;
                exact_oscillator_step(part.x[jj], part.velocity[jj], equilibrium,
                                     omega2, mu, c.dt, x_next, v_next);
                apply_clamped_restitution(x_next, v_next, lower[jj], upper[jj], c.boundary_restitution);
                part.x[jj] = x_next;
                part.velocity[jj] = v_next;
            }
            part.value = evaluate_benchmark(c.benchmark, part.x, optimum);
            if (part.value < part.pbest_value) {
                part.pbest_value = part.value;
                part.pbest = part.x;
            }
        }
        compute_seconds += std::chrono::duration<double>(clock::now() - compute_start).count();
        out.function_evaluations += static_cast<uint64_t>(n);

        // Deterministic island reductions. All objective values are evaluated
        // concurrently; only the small leader records are reduced serially.
        const auto reduction_start = clock::now();
        std::vector<Leader> local_leaders(static_cast<std::size_t>(p));
        for (int i = 0; i < n; ++i) {
            const Particle &part = particles[static_cast<std::size_t>(i)];
            const int island = island_of_particle[static_cast<std::size_t>(i)];
            Leader &local = local_leaders[static_cast<std::size_t>(island)];
            if (part.pbest_value < local.value) {
                local.value = part.pbest_value;
                local.position = part.pbest;
            }
        }
        for (int k = 0; k < p; ++k) {
            leaders[static_cast<std::size_t>(k)] =
                better_of(leaders[static_cast<std::size_t>(k)], local_leaders[static_cast<std::size_t>(k)]);
        }

        const auto best_now = global_leader();
        out.history.push_back(std::min(out.history.back(), best_now.value));
        communication_seconds += std::chrono::duration<double>(clock::now() - reduction_start).count();

        if (((iter + 1) % sync_interval) == 0 || (iter + 1) == c.iterations) {
            const auto comm_start = clock::now();
            if (c.topology == CPSO_TOPOLOGY_STAR) {
                const Leader common = global_leader();
                for (int k = 0; k < p; ++k) targets[static_cast<std::size_t>(k)] = common;
            } else if (c.topology == CPSO_TOPOLOGY_RING && p > 1) {
                // Synchronous one-hop ring diffusion. The snapshot prevents
                // order-dependent propagation within a single communication event.
                const std::vector<Leader> snapshot = leaders;
                for (int k = 0; k < p; ++k) {
                    const int left = (k + p - 1) % p;
                    targets[static_cast<std::size_t>(k)] =
                        better_of(snapshot[static_cast<std::size_t>(k)], snapshot[static_cast<std::size_t>(left)]);
                }
                // A received incumbent is retained as the island's leader and
                // can therefore propagate one edge at each subsequent event.
                for (int k = 0; k < p; ++k)
                    leaders[static_cast<std::size_t>(k)] = better_of(leaders[static_cast<std::size_t>(k)], targets[static_cast<std::size_t>(k)]);
            } else {
                for (int k = 0; k < p; ++k) targets[static_cast<std::size_t>(k)] = leaders[static_cast<std::size_t>(k)];
            }
            communication_seconds += std::chrono::duration<double>(clock::now() - comm_start).count();
        }
    }

    const Leader final_best = global_leader();
    out.best_value = final_best.value;
    out.best_position = final_best.position;
    out.worker_best_values.resize(static_cast<std::size_t>(p));
    for (int k = 0; k < p; ++k)
        out.worker_best_values[static_cast<std::size_t>(k)] = leaders[static_cast<std::size_t>(k)].value;
    out.compute_seconds = compute_seconds;
    out.communication_seconds = communication_seconds;
    out.elapsed_seconds = std::chrono::duration<double>(clock::now() - total_start).count();
    return out;
}
} // namespace cpso
