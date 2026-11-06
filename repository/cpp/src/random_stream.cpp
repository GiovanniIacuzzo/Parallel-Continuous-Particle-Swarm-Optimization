#include "cpso/random_stream.hpp"

#include <cstdint>
#include <stdexcept>

namespace cpso {

std::mt19937_64 make_particle_random_engine(
    const std::uint64_t run_seed,
    const std::uint64_t particle_id) {
    if (particle_id == 0U) {
        throw std::invalid_argument("particle_id must be one-based and positive");
    }

    const std::uint32_t seed_low = static_cast<std::uint32_t>(run_seed & 0xFFFFFFFFULL);
    const std::uint32_t seed_high = static_cast<std::uint32_t>(run_seed >> 32U);
    const std::uint32_t particle_low = static_cast<std::uint32_t>(particle_id & 0xFFFFFFFFULL);
    const std::uint32_t particle_high = static_cast<std::uint32_t>(particle_id >> 32U);

    std::seed_seq material{
        seed_low,
        seed_high,
        particle_low,
        particle_high,
        0x4350534FU,  // "CPSO"
        0x5354524DU,  // "STRM"
    };
    std::mt19937_64 engine;
    engine.seed(material);
    return engine;
}

}  // namespace cpso
