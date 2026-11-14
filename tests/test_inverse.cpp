#include "cpso/inverse_diffusion_benchmark.hpp"
#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<double> truth(30), shifted(30);
    for (size_t j = 0; j < truth.size(); ++j) {
        truth[j] = 0.4 * std::sin(2.0 * pi * static_cast<double>(j + 1U)
                                   / static_cast<double>(truth.size() + 1U));
        shifted[j] = truth[j] + 0.35;
    }
    for (std::uint64_t steps : {16ULL, 128ULL, 512ULL}) {
        double optimal = cpso::evaluate_inverse_diffusion_benchmark(truth, steps);
        double other = cpso::evaluate_inverse_diffusion_benchmark(shifted, steps);
        assert(optimal == 0.0 && other > 0.0 && std::isfinite(other));
        assert(other == cpso::evaluate_inverse_diffusion_benchmark(shifted, steps));
    }
    const auto baseline = cpso::evaluate_inverse_diffusion_benchmark(shifted, 256);
    std::vector<std::future<double>> futures;
    for (int k=0;k<8;k++) futures.push_back(std::async(std::launch::async,[&] {
        return cpso::evaluate_inverse_diffusion_benchmark(shifted,256);
    }));
    for(auto &f:futures) assert(f.get()==baseline);
    bool bad=false;
    try { (void)cpso::evaluate_inverse_diffusion_benchmark(shifted, 17); }
    catch (const std::invalid_argument&) { bad=true; }
    assert(bad);
    std::cout << "Inverse diffusion RK4: optimum, determinism, concurrency, errors PASS\n";
}
