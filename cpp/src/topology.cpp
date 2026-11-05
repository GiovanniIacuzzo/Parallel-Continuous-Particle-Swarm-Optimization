#include "cpso/topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cpso {
namespace {

void sort_unique(std::vector<std::size_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

}  // namespace

ParticleTopology::ParticleTopology(
    const TopologyKind kind,
    const std::size_t population_size,
    const std::size_t grid_rows,
    const std::size_t grid_columns)
    : kind_(kind), population_size_(population_size) {
    if (population_size == 0U) {
        throw std::invalid_argument("Topology requires population_size >= 1");
    }

    switch (kind_) {
        case TopologyKind::GlobalBest: {
            if (grid_rows != 0U || grid_columns != 0U) {
                throw std::invalid_argument("Grid dimensions are only valid for grid topology");
            }
            global_neighbors_.resize(population_size);
            for (std::size_t i = 0; i < population_size; ++i) {
                global_neighbors_[i] = i;
            }
            break;
        }
        case TopologyKind::Ring: {
            if (grid_rows != 0U || grid_columns != 0U) {
                throw std::invalid_argument("Grid dimensions are only valid for grid topology");
            }
            neighborhoods_.resize(population_size);
            for (std::size_t i = 0; i < population_size; ++i) {
                auto& neighborhood = neighborhoods_[i];
                neighborhood = {
                    (i + population_size - 1U) % population_size,
                    i,
                    (i + 1U) % population_size,
                };
                sort_unique(neighborhood);
            }
            break;
        }
        case TopologyKind::GridVonNeumann: {
            if (grid_rows < 2U || grid_columns < 2U) {
                throw std::invalid_argument("Grid topology requires grid_rows >= 2 and grid_columns >= 2");
            }
            if (grid_rows > std::numeric_limits<std::size_t>::max() / grid_columns
                || grid_rows * grid_columns != population_size) {
                throw std::invalid_argument("grid_rows * grid_columns must equal population_size");
            }
            neighborhoods_.resize(population_size);
            for (std::size_t index = 0; index < population_size; ++index) {
                const std::size_t row = index / grid_columns;
                const std::size_t column = index % grid_columns;
                const std::size_t up = ((row + grid_rows - 1U) % grid_rows) * grid_columns + column;
                const std::size_t down = ((row + 1U) % grid_rows) * grid_columns + column;
                const std::size_t left = row * grid_columns + ((column + grid_columns - 1U) % grid_columns);
                const std::size_t right = row * grid_columns + ((column + 1U) % grid_columns);
                auto& neighborhood = neighborhoods_[index];
                neighborhood = {index, up, down, left, right};
                sort_unique(neighborhood);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported particle topology");
    }
}

const std::vector<std::size_t>& ParticleTopology::neighbors(
    const std::size_t particle_index) const {
    if (particle_index >= population_size_) {
        throw std::out_of_range("Particle topology index out of range");
    }
    if (kind_ == TopologyKind::GlobalBest) {
        return global_neighbors_;
    }
    return neighborhoods_[particle_index];
}

std::vector<std::size_t> ParticleTopology::select_local_leaders(
    const std::vector<double>& personal_best_values) const {
    if (kind_ == TopologyKind::GlobalBest) {
        throw std::logic_error("Global-best leader selection uses the incumbent global-best index");
    }
    if (personal_best_values.size() != population_size_) {
        throw std::invalid_argument("Personal-best value count does not match topology population");
    }
    for (const double value : personal_best_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Personal-best values must be finite");
        }
    }

    std::vector<std::size_t> leaders(population_size_, 0U);
    for (std::size_t i = 0; i < population_size_; ++i) {
        const auto& neighborhood = neighborhoods_[i];
        std::size_t leader = neighborhood.front();
        for (const std::size_t candidate : neighborhood) {
            if (personal_best_values[candidate] < personal_best_values[leader]
                || (personal_best_values[candidate] == personal_best_values[leader]
                    && candidate < leader)) {
                leader = candidate;
            }
        }
        leaders[i] = leader;
    }
    return leaders;
}

}  // namespace cpso
