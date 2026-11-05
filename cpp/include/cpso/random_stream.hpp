#pragma once

#include <cstdint>
#include <random>

namespace cpso {

// Construct a deterministic per-particle random stream from two separate
// identifiers. Using seed_seq avoids the simple deterministic pre-hash
// aliases that occur when combining run_seed and particle_id with XOR.
[[nodiscard]] std::mt19937_64 make_particle_random_engine(
    std::uint64_t run_seed,
    std::uint64_t particle_id);

}  // namespace cpso
