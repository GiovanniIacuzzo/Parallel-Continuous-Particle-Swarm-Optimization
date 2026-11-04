#pragma once

namespace cpso {

struct OscillatorState {
    double position{};
    double velocity{};
};

// Exact flow of the frozen scalar-coefficient ODE
//   x'' + (1 - mu) x' + omega2 * x = omega2 * equilibrium.
// The returned flow is exact up to floating-point roundoff for the supplied
// constant coefficients. It does not make a sequence of refreshed coefficients
// an exact solution of one global, time-varying ODE.
[[nodiscard]] OscillatorState propagate_frozen_oscillator(
    double position,
    double velocity,
    double equilibrium,
    double omega2,
    double mu,
    double dt);

// Explicit algorithmic boundary map used by the serial optimizer. If a proposed
// coordinate crosses a bound, it is clamped at that bound and its outward
// velocity is reversed and multiplied by restitution in [0,1].
void apply_clamped_restitution(
    double& position,
    double& velocity,
    double lower,
    double upper,
    double restitution);

}  // namespace cpso
