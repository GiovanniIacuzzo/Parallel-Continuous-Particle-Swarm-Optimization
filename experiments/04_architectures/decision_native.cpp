#include "cpso/optimizer.hpp"
#include "cpso/objectives.hpp"
#include "cpso/inverse_diffusion_benchmark.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdexcept>

// Observable side effect outside the objective value, private to each worker.
static thread_local volatile double work_sink;

int main(int argc, char** argv) {
    try {
        if (argc != 10) throw std::invalid_argument("expected obj D graph swarm_seed graph_seed backend workers work iterations");
        const std::string obj = argv[1], graph = argv[3], backend = argv[6];
        const auto dim = std::stoul(argv[2]); const auto graph_seed = std::stoull(argv[5]);
        const auto workers = std::stoul(argv[7]), work = std::stoul(argv[8]);
        cpso::Config cfg;
        cfg.seed = std::stoull(argv[4]);
        cfg.population_size = 64;
        cfg.iterations = std::stoul(argv[9]);
        cfg.dt = 1.;
        double bound = obj == "rastrigin" ? 5.12 : 2.;
        cfg.lower_bounds.assign(dim, -bound);
        cfg.upper_bounds.assign(dim, bound);
        if (graph == "grid") {
            cfg.topology = cpso::TopologyKind::GridVonNeumann;
            cfg.grid_rows = cfg.grid_columns = 8;
        } else if (graph == "random4" || graph == "random8") {
            cfg.topology = cpso::TopologyKind::RandomConnected;
            cfg.topology_degree = graph == "random4" ? 4 : 8;
        } else if (graph != "global") throw std::invalid_argument("unknown graph");
        cpso::Objective objective;
        if (obj == "diffusion") {
            objective = [](const auto& x){return cpso::evaluate_inverse_diffusion_benchmark(x, 256);};
        } else if (obj == "rastrigin") {
            objective = [work](const auto& x) {
                const auto value = cpso::evaluate_benchmark(cpso::Benchmark::Rastrigin, x);
                if (work) {
                    double state = x.front();
                    for (std::size_t k=0; k<work; ++k)
                        state = std::fma(state, 0.999999, double(k % 31) * 1.e-6);
                    work_sink = state;
                }
                return value;
            };
        } else throw std::invalid_argument("unknown objective");
        cpso::Result r;
        if (backend == "serial") r = cpso::minimize_serial(cfg,objective);
        else if (backend == "master") r = cpso::minimize_master_slave(cfg,objective,workers);
        else if (backend == "particle") r = cpso::minimize_particle_parallel(cfg,objective,workers,graph_seed);
        else throw std::invalid_argument("unknown backend");
        cpso::ParticleTopology g(cfg.topology,64,cfg.grid_rows,cfg.grid_columns,graph_seed,cfg.topology_degree);
        double degree = 0.;
        for (std::size_t j=0;j<64;++j) degree += (g.neighbors(j).size()-1)/64.;
        std::cout << std::setprecision(17) << r.elapsed_seconds << ' ' << r.evaluation_wall_seconds
                  << ' ' << r.function_evaluations << ' ' << degree << '\n';
        for (double v:r.best_history) std::cout << v << ' ';
        std::cout << '\n';
        for (double v:r.best_position) std::cout << v << ' ';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
