#pragma once

#include <cstddef>
#include <vector>

namespace cpso {

// Social topology used to select the attractor for each particle.
enum class TopologyKind : int {
    GlobalBest = 0,
    Ring = 1,
    GridVonNeumann = 2,
};

// Immutable neighborhood graph. Every neighborhood includes its own particle
// and is sorted by particle index, with duplicates removed.
class ParticleTopology {
public:
    ParticleTopology(
        TopologyKind kind,
        std::size_t population_size,
        std::size_t grid_rows = 0,
        std::size_t grid_columns = 0);

    [[nodiscard]] TopologyKind kind() const noexcept { return kind_; }
    [[nodiscard]] std::size_t population_size() const noexcept { return population_size_; }
    [[nodiscard]] const std::vector<std::size_t>& neighbors(std::size_t particle_index) const;

    // Select the best personal-best index in each neighborhood. Ties are
    // resolved by the smallest particle index. This function is intended for
    // ring/grid; global-best keeps its incumbent index to preserve the exact
    // legacy gbest tie/update semantics.
    [[nodiscard]] std::vector<std::size_t> select_local_leaders(
        const std::vector<double>& personal_best_values) const;

private:
    TopologyKind kind_;
    std::size_t population_size_{};
    // Global-best has one shared O(N) index list rather than N duplicated
    // copies (which would consume O(N^2) memory).
    std::vector<std::size_t> global_neighbors_;
    // Ring and grid store one small neighborhood per particle.
    std::vector<std::vector<std::size_t>> neighborhoods_;
};

}  // namespace cpso
