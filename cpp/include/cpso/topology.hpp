#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cpso {

// Social topology used to select the attractor for each particle.
enum class TopologyKind : int {
    GlobalBest = 0,
    Ring = 1,
    GridVonNeumann = 2,
    Wheel = 3,
    RandomConnected = 4,
    SmallWorld = 5,
    ScaleFree = 6,
};

// Immutable, undirected neighborhood graph. Every neighborhood includes its
// own particle and is sorted by particle index, with duplicates removed.
// Randomized graph families are generated once per run from topology_seed.
class ParticleTopology {
public:
    ParticleTopology(
        TopologyKind kind,
        std::size_t population_size,
        std::size_t grid_rows = 0,
        std::size_t grid_columns = 0,
        std::uint64_t topology_seed = 1,
        std::size_t topology_degree = 4,
        double rewiring_probability = 0.15);

    [[nodiscard]] TopologyKind kind() const noexcept { return kind_; }
    [[nodiscard]] std::size_t population_size() const noexcept { return population_size_; }
    [[nodiscard]] const std::vector<std::size_t>& neighbors(std::size_t particle_index) const;

    // Select the best personal-best index in each neighborhood. Ties are
    // resolved by the smallest particle index. Global-best keeps its incumbent
    // index to preserve the legacy gbest tie/update semantics.
    [[nodiscard]] std::vector<std::size_t> select_local_leaders(
        const std::vector<double>& personal_best_values) const;

private:
    TopologyKind kind_;
    std::size_t population_size_{};
    std::vector<std::size_t> global_neighbors_;
    std::vector<std::vector<std::size_t>> neighborhoods_;
};

}  // namespace cpso
