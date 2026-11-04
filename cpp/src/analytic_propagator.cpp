#include "cpso/analytic_propagator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cpso {
namespace {

void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

}  // namespace

OscillatorState propagate_frozen_oscillator(
    const double position,
    const double velocity,
    const double equilibrium,
    const double omega2,
    const double mu,
    const double dt) {
    require_finite(position, "position");
    require_finite(velocity, "velocity");
    require_finite(equilibrium, "equilibrium");
    require_finite(omega2, "omega2");
    require_finite(mu, "mu");
    require_finite(dt, "dt");

    if (omega2 < 0.0) {
        throw std::invalid_argument("omega2 must be non-negative");
    }
    if (mu < 0.0 || mu > 1.0) {
        throw std::invalid_argument("mu must lie in [0,1]");
    }
    if (dt < 0.0) {
        throw std::invalid_argument("dt must be non-negative");
    }
    if (dt == 0.0) {
        return {position, velocity};
    }

    // alpha = (1-mu)/2, so the ODE is
    // u'' + 2 alpha u' + omega^2 u = 0, where u = x - equilibrium.
    const double alpha = 0.5 * (1.0 - mu);
    const double discriminant = alpha * alpha - omega2;
    const double u0 = position - equilibrium;
    const double scale = std::max({1.0, alpha * alpha, omega2});
    const double critical_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;

    OscillatorState next{};
    if (std::abs(discriminant) <= critical_tolerance) {
        // Critically damped (repeated characteristic root -alpha).
        const double c2 = velocity + alpha * u0;
        const double decay = std::exp(-alpha * dt);
        const double q = u0 + c2 * dt;
        next.position = equilibrium + decay * q;
        next.velocity = decay * (velocity - alpha * c2 * dt);
    } else if (discriminant < 0.0) {
        // Underdamped real trigonometric representation.
        const double beta = std::sqrt(-discriminant);
        const double c = (velocity + alpha * u0) / beta;
        const double angle = beta * dt;
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const double q = u0 * cosine + c * sine;
        const double dq = -u0 * beta * sine + c * beta * cosine;
        const double decay = std::exp(-alpha * dt);
        next.position = equilibrium + decay * q;
        next.velocity = decay * (dq - alpha * q);
    } else {
        // Overdamped: two distinct real characteristic roots.
        const double root = std::sqrt(discriminant);
        const double lambda1 = -alpha + root;
        const double lambda2 = -alpha - root;
        const double denominator = lambda1 - lambda2;
        const double c1 = (velocity - lambda2 * u0) / denominator;
        const double c2 = (lambda1 * u0 - velocity) / denominator;
        const double e1 = std::exp(lambda1 * dt);
        const double e2 = std::exp(lambda2 * dt);
        next.position = equilibrium + c1 * e1 + c2 * e2;
        next.velocity = c1 * lambda1 * e1 + c2 * lambda2 * e2;
    }

    if (!std::isfinite(next.position) || !std::isfinite(next.velocity)) {
        throw std::overflow_error("Frozen oscillator propagation produced a non-finite state");
    }
    return next;
}

void apply_clamped_restitution(
    double& position,
    double& velocity,
    const double lower,
    const double upper,
    const double restitution) {
    if (!std::isfinite(position) || !std::isfinite(velocity)
        || !std::isfinite(lower) || !std::isfinite(upper)
        || !std::isfinite(restitution)) {
        throw std::invalid_argument("Boundary-map inputs must be finite");
    }
    if (!(lower < upper)) {
        throw std::invalid_argument("Boundary interval must satisfy lower < upper");
    }
    if (restitution < 0.0 || restitution > 1.0) {
        throw std::invalid_argument("restitution must lie in [0,1]");
    }

    if (position < lower) {
        position = lower;
        if (velocity < 0.0) {
            velocity = -restitution * velocity;
        }
    } else if (position > upper) {
        position = upper;
        if (velocity > 0.0) {
            velocity = -restitution * velocity;
        }
    }
}

}  // namespace cpso
