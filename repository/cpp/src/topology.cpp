#include "cpso/topology.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace cpso {
namespace {
using Graph = std::vector<std::set<std::size_t>>;

void add_edge(Graph& graph, const std::size_t a, const std::size_t b) {
    if (a == b) {
        return;
    }
    graph[a].insert(b);
    graph[b].insert(a);
}

void remove_edge(Graph& graph, const std::size_t a, const std::size_t b) {
    graph[a].erase(b);
    graph[b].erase(a);
}

std::uint64_t mix_seed(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

void validate_non_grid_parameters(
    const std::size_t grid_rows,
    const std::size_t grid_columns) {
    if (grid_rows != 0U || grid_columns != 0U) {
        throw std::invalid_argument("Grid dimensions are only valid for grid topology");
    }
}

std::vector<std::vector<std::size_t>> connected_components(const Graph& graph) {
    const std::size_t n = graph.size();
    std::vector<bool> visited(n, false);
    std::vector<std::vector<std::size_t>> components;
    for (std::size_t start = 0; start < n; ++start) {
        if (visited[start]) {
            continue;
        }
        components.emplace_back();
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = true;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            components.back().push_back(node);
            for (const std::size_t neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
    }
    return components;
}

Graph make_wheel(const std::size_t n, std::mt19937_64& rng) {
    if (n < 2U) {
        throw std::invalid_argument("Wheel topology requires population_size >= 2");
    }
    // Randomly assign graph roles to particle IDs. This prevents particle 0
    // from being the hub in every seed while preserving the wheel graph.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0U);
    std::shuffle(order.begin(), order.end(), rng);

    Graph graph(n);
    const std::size_t hub = order.front();
    for (std::size_t index = 1U; index < n; ++index) {
        // Canonical PSO wheel: all information is mediated by one focal
        // particle; peripheral particles do not communicate directly.
        add_edge(graph, hub, order[index]);
    }
    return graph;
}

Graph make_random_connected(
    const std::size_t n,
    const std::size_t target_degree,
    std::mt19937_64& rng) {
    if (n < 3U || target_degree == 0U || target_degree >= n) {
        throw std::invalid_argument("Random topology requires N >= 3 and 1 <= topology_degree < N");
    }
    Graph graph(n);
    const double probability = static_cast<double>(target_degree)
        / static_cast<double>(n - 1U);
    std::bernoulli_distribution include_edge(probability);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1U; j < n; ++j) {
            if (include_edge(rng)) {
                add_edge(graph, i, j);
            }
        }
    }

    // A connected communication graph avoids isolated sub-swarms. We first
    // sample G(N,p), then connect its components by random inter-component
    // bridges. This is explicitly a connected-ER construction, not an exact
    // sample from G(N,p) conditioned on connectivity.
    while (true) {
        const auto components = connected_components(graph);
        if (components.size() <= 1U) {
            break;
        }
        std::uniform_int_distribution<std::size_t> component_pick(0U, components.size() - 1U);
        const std::size_t first_component = component_pick(rng);
        std::size_t second_component = component_pick(rng);
        while (second_component == first_component) {
            second_component = component_pick(rng);
        }
        std::uniform_int_distribution<std::size_t> first_node_pick(
            0U, components[first_component].size() - 1U);
        std::uniform_int_distribution<std::size_t> second_node_pick(
            0U, components[second_component].size() - 1U);
        add_edge(
            graph,
            components[first_component][first_node_pick(rng)],
            components[second_component][second_node_pick(rng)]);
    }
    return graph;
}

Graph make_small_world(
    const std::size_t n,
    const std::size_t degree,
    const double rewiring_probability,
    std::mt19937_64& rng) {
    if (n < 4U || degree < 2U || degree >= n || degree % 2U != 0U) {
        throw std::invalid_argument("Small-world topology requires even 2 <= topology_degree < N and N >= 4");
    }
    if (!std::isfinite(rewiring_probability)
        || rewiring_probability < 0.0 || rewiring_probability > 1.0) {
        throw std::invalid_argument("rewiring_probability must lie in [0,1]");
    }

    Graph graph(n);
    const std::size_t half_degree = degree / 2U;
    for (std::size_t node = 0; node < n; ++node) {
        for (std::size_t offset = 1U; offset <= half_degree; ++offset) {
            add_edge(graph, node, (node + offset) % n);
        }
    }

    std::bernoulli_distribution rewire(rewiring_probability);
    for (std::size_t node = 0; node < n; ++node) {
        for (std::size_t offset = 1U; offset <= half_degree; ++offset) {
            const std::size_t old_target = (node + offset) % n;
            if (!graph[node].contains(old_target) || !rewire(rng)) {
                continue;
            }
            std::vector<std::size_t> candidates;
            candidates.reserve(n);
            for (std::size_t candidate = 0; candidate < n; ++candidate) {
                if (candidate != node && !graph[node].contains(candidate)) {
                    candidates.push_back(candidate);
                }
            }
            if (candidates.empty()) {
                continue;
            }
            std::uniform_int_distribution<std::size_t> pick(0U, candidates.size() - 1U);
            remove_edge(graph, node, old_target);
            add_edge(graph, node, candidates[pick(rng)]);
        }
    }
    return graph;
}

Graph make_scale_free(
    const std::size_t n,
    const std::size_t degree_parameter,
    std::mt19937_64& rng) {
    if (degree_parameter < 2U || degree_parameter % 2U != 0U) {
        throw std::invalid_argument("Scale-free topology requires an even topology_degree >= 2");
    }
    const std::size_t m = degree_parameter / 2U;
    if (n < m + 1U) {
        throw std::invalid_argument("Scale-free topology requires N >= topology_degree/2 + 1");
    }

    Graph graph(n);
    const std::size_t initial_nodes = m + 1U;
    for (std::size_t i = 0; i < initial_nodes; ++i) {
        for (std::size_t j = i + 1U; j < initial_nodes; ++j) {
            add_edge(graph, i, j);
        }
    }

    for (std::size_t node = initial_nodes; node < n; ++node) {
        std::vector<std::size_t> selected;
        selected.reserve(m);
        for (std::size_t edge = 0; edge < m; ++edge) {
            std::uint64_t total_degree = 0U;
            for (std::size_t candidate = 0; candidate < node; ++candidate) {
                if (std::find(selected.begin(), selected.end(), candidate) == selected.end()) {
                    total_degree += static_cast<std::uint64_t>(graph[candidate].size());
                }
            }
            if (total_degree == 0U) {
                throw std::logic_error("Scale-free preferential attachment encountered zero total degree");
            }
            std::uniform_int_distribution<std::uint64_t> pick(0U, total_degree - 1U);
            std::uint64_t ticket = pick(rng);
            std::size_t chosen = 0U;
            for (; chosen < node; ++chosen) {
                if (std::find(selected.begin(), selected.end(), chosen) != selected.end()) {
                    continue;
                }
                const auto weight = static_cast<std::uint64_t>(graph[chosen].size());
                if (ticket < weight) {
                    break;
                }
                ticket -= weight;
            }
            if (chosen >= node) {
                throw std::logic_error("Scale-free preferential attachment failed to select a node");
            }
            selected.push_back(chosen);
            add_edge(graph, node, chosen);
        }
    }
    // Barabasi-Albert growth gives early vertices high expected degree. Relabel
    // the completed graph randomly so high-degree roles are not tied to fixed
    // particle IDs (whose random streams are themselves indexed by particle).
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0U);
    std::shuffle(order.begin(), order.end(), rng);
    Graph relabelled(n);
    for (std::size_t node = 0; node < n; ++node) {
        for (const std::size_t neighbor : graph[node]) {
            if (node < neighbor) {
                add_edge(relabelled, order[node], order[neighbor]);
            }
        }
    }
    return relabelled;
}

std::vector<std::vector<std::size_t>> finalize_graph(const Graph& graph) {
    std::vector<std::vector<std::size_t>> neighborhoods(graph.size());
    for (std::size_t i = 0; i < graph.size(); ++i) {
        neighborhoods[i].reserve(graph[i].size() + 1U);
        neighborhoods[i].push_back(i);
        neighborhoods[i].insert(neighborhoods[i].end(), graph[i].begin(), graph[i].end());
        std::sort(neighborhoods[i].begin(), neighborhoods[i].end());
        neighborhoods[i].erase(
            std::unique(neighborhoods[i].begin(), neighborhoods[i].end()),
            neighborhoods[i].end());
    }
    return neighborhoods;
}

}  // namespace

ParticleTopology::ParticleTopology(
    const TopologyKind kind,
    const std::size_t population_size,
    const std::size_t grid_rows,
    const std::size_t grid_columns,
    const std::uint64_t topology_seed,
    const std::size_t topology_degree,
    const double rewiring_probability)
    : kind_(kind), population_size_(population_size) {
    if (population_size == 0U) {
        throw std::invalid_argument("Topology requires population_size >= 1");
    }
    std::mt19937_64 rng(mix_seed(topology_seed));

    switch (kind_) {
        case TopologyKind::GlobalBest: {
            validate_non_grid_parameters(grid_rows, grid_columns);
            global_neighbors_.resize(population_size);
            for (std::size_t i = 0; i < population_size; ++i) {
                global_neighbors_[i] = i;
            }
            break;
        }
        case TopologyKind::Ring: {
            validate_non_grid_parameters(grid_rows, grid_columns);
            neighborhoods_.resize(population_size);
            for (std::size_t i = 0; i < population_size; ++i) {
                auto& neighborhood = neighborhoods_[i];
                neighborhood = {
                    (i + population_size - 1U) % population_size,
                    i,
                    (i + 1U) % population_size,
                };
                std::sort(neighborhood.begin(), neighborhood.end());
                neighborhood.erase(std::unique(neighborhood.begin(), neighborhood.end()), neighborhood.end());
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
                std::sort(neighborhood.begin(), neighborhood.end());
                neighborhood.erase(std::unique(neighborhood.begin(), neighborhood.end()), neighborhood.end());
            }
            break;
        }
        case TopologyKind::Wheel:
            validate_non_grid_parameters(grid_rows, grid_columns);
            neighborhoods_ = finalize_graph(make_wheel(population_size, rng));
            break;
        case TopologyKind::RandomConnected:
            validate_non_grid_parameters(grid_rows, grid_columns);
            neighborhoods_ = finalize_graph(make_random_connected(population_size, topology_degree, rng));
            break;
        case TopologyKind::SmallWorld:
            validate_non_grid_parameters(grid_rows, grid_columns);
            neighborhoods_ = finalize_graph(make_small_world(
                population_size, topology_degree, rewiring_probability, rng));
            break;
        case TopologyKind::ScaleFree:
            validate_non_grid_parameters(grid_rows, grid_columns);
            neighborhoods_ = finalize_graph(make_scale_free(population_size, topology_degree, rng));
            break;
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
