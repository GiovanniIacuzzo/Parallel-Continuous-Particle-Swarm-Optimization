#pragma once

#include "cpso/analytic_propagator.hpp"
#include "cpso/optimizer.hpp"
#include "cpso/random_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace cpso::detail {

using Clock = std::chrono::steady_clock;

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> personal_best;
    double current_value{std::numeric_limits<double>::infinity()};
    double personal_best_value{std::numeric_limits<double>::infinity()};
    std::mt19937_64 random_engine;
};

inline double uniform01(std::mt19937_64& engine) {
    return std::generate_canonical<double, 53>(engine);
}

inline double inertia_at(const Config& config, const std::size_t iteration) {
    if (config.iterations <= 1U) {
        return config.mu_max;
    }
    const double fraction = static_cast<double>(iteration)
        / static_cast<double>(config.iterations - 1U);
    return config.mu_max - (config.mu_max - config.mu_min) * fraction;
}

inline double checked_evaluate(
    const Objective& objective,
    const std::vector<double>& position) {
    const double value = objective(position);
    if (!std::isfinite(value)) {
        throw std::runtime_error("Objective function returned a non-finite value");
    }
    return value;
}

inline void initialize_particle(
    Particle& particle,
    const Config& config,
    const std::size_t dimension,
    const std::uint64_t one_based_particle_id) {
    particle.random_engine = make_particle_random_engine(config.seed, one_based_particle_id);
    particle.position.resize(dimension);
    particle.velocity.resize(dimension);
    particle.personal_best.resize(dimension);
    for (std::size_t j = 0; j < dimension; ++j) {
        const double lower = config.lower_bounds[j];
        const double upper = config.upper_bounds[j];
        const double width = upper - lower;
        particle.position[j] = lower + uniform01(particle.random_engine) * width;
        particle.velocity[j] = (2.0 * uniform01(particle.random_engine) - 1.0)
                               * config.initial_velocity_fraction * width;
    }
}

inline void propagate_particle(
    Particle& particle,
    const std::vector<double>& social_attractor,
    const Config& config,
    const std::size_t iteration,
    std::vector<double>& next_position,
    std::vector<double>& next_velocity) {
    // The thesis writes r1,r2 as scalar random variables: draw one pair per
    // particle and frozen interval, then share it across all coordinates.
    const double r1 = uniform01(particle.random_engine);
    const double r2 = uniform01(particle.random_engine);
    const double cognitive_weight = config.cognitive * r1;
    const double social_weight = config.social * r2;
    const double omega2 = cognitive_weight + social_weight;
    // mu is the inertia weight w_k in the thesis ODE coefficient (1 - w_k).
    const double mu = inertia_at(config, iteration);
    const std::size_t dimension = config.lower_bounds.size();

    for (std::size_t j = 0; j < dimension; ++j) {
        double equilibrium = 0.0;
        if (omega2 > 0.0) {
            equilibrium = (cognitive_weight * particle.personal_best[j]
                           + social_weight * social_attractor[j]) / omega2;
        }
        OscillatorState next = propagate_frozen_oscillator(
            particle.position[j], particle.velocity[j], equilibrium,
            omega2, mu, config.dt);
        apply_clamped_restitution(
            next.position, next.velocity,
            config.lower_bounds[j], config.upper_bounds[j],
            config.boundary_restitution);
        next_position[j] = next.position;
        next_velocity[j] = next.velocity;
    }
}

inline std::size_t global_best_index(const std::vector<Particle>& particles) {
    std::size_t best = 0U;
    for (std::size_t i = 1U; i < particles.size(); ++i) {
        if (particles[i].personal_best_value < particles[best].personal_best_value) {
            best = i;
        }
    }
    return best;
}

inline std::vector<std::vector<double>> local_social_snapshot(
    const std::vector<Particle>& particles,
    const ParticleTopology& topology) {
    std::vector<double> values(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        values[i] = particles[i].personal_best_value;
    }
    const std::vector<std::size_t> leaders = topology.select_local_leaders(values);
    std::vector<std::vector<double>> snapshot(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        snapshot[i] = particles[leaders[i]].personal_best;
    }
    return snapshot;
}

}  // namespace cpso::detail
