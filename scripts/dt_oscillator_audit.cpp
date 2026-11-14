// Independent RK4 numerical cross-check for the analytic frozen oscillator.
// Build from the repository root:
// c++ -std=c++20 -O2 -Icpp/include cpp/src/analytic_propagator.cpp \
//   scripts/dt_oscillator_audit.cpp -o build-m2/dt_oscillator_audit
#include "cpso/analytic_propagator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

struct State { double x,v; };

State rk4(double x0,double v0,double q,double omega2,double mu,double dt) {
    int steps=std::max(2000,static_cast<int>(std::ceil(dt*2500)));
    double h=dt/steps;
    State s{x0,v0};
    const auto rate=[&](State y)->State{
        return {y.v,-(1-mu)*y.v-omega2*(y.x-q)};
    };
    const auto add=[&](State a,State b,double scale)->State{
        return {a.x+scale*b.x,a.v+scale*b.v};
    };
    for(int i=0;i<steps;++i) {
        State k1=rate(s);
        State k2=rate(add(s,k1,h/2));
        State k3=rate(add(s,k2,h/2));
        State k4=rate(add(s,k3,h));
        s.x+=h*(k1.x+2*k2.x+2*k3.x+k4.x)/6;
        s.v+=h*(k1.v+2*k2.v+2*k3.v+k4.v)/6;
    }
    return s;
}

int main() {
    // mu=0.4 => alpha=(1-mu)/2=0.3, discriminant=alpha^2-omega2.
    const struct { double omega2,mu,dt; } cases[] = {
        {2.05,0.9,0.01}, {2.05,0.9,0.2}, {2.05,0.4,30.0},
        {0.09,0.4,0.5}, {0.09,0.4,30.0}, // critical damping
        {0.01,0.4,0.5}, {0.01,0.4,30.0}, // overdamped
        {4.1,0.4,3.6}, {4.1,0.4,30.0}
    };
    for (const auto c:cases) {
        const double x0=12.345, v0=-1.23,q=-2.5;
        auto a=cpso::propagate_frozen_oscillator(x0,v0,q,c.omega2,c.mu,c.dt);
        auto n=rk4(x0,v0,q,c.omega2,c.mu,c.dt);
        double error=std::max(std::abs(a.position-n.x),std::abs(a.velocity-n.v));
        if (!std::isfinite(error) || error>2e-7) {
            std::cerr<<"FAIL omega2="<<c.omega2<<" mu="<<c.mu
                     <<" dt="<<c.dt<<" abs error="<<error<<"\n";
            return 1;
        }
    }
    std::cout<<"PASS: nine analytic oscillator cases match independent RK4 reference\n";
    return 0;
}
