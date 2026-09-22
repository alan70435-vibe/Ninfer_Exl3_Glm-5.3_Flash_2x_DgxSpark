#include "ninfer_glm53/exl3_decode.hpp"
#include "exl3_test_oracle.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>

static void expect(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
int main() {
    using namespace ninfer::glm53;
    std::uint16_t packed[64], states[256], perm[256];
    exl3_tensor_core_perm(perm);
    bool seen[256]{};
    for (unsigned i=0;i<256;++i) {
        expect(perm[i] < 256 && !seen[perm[i]], "permutation not bijective");
        seen[perm[i]] = true;
        expect(oracle_order(perm[i]/16,perm[i]%16) == i, "permutation differs from scalar layout");
    }
    for (unsigned state=0;state<65536;++state)
        expect(mcg_symbol(state) == oracle_mcg(static_cast<std::uint16_t>(state)), "MCG binary16 rounding");
    for (unsigned pattern=0;pattern<10;++pattern) {
        for (unsigned i=0;i<64;++i)
            packed[i] = pattern == 0 ? 0 : pattern == 1 ? 65535 : static_cast<std::uint16_t>(i*40503u+pattern*32771u);
        unpack_trellis_k4(packed,states);
        float tile[256]; exl3_decode_inner_tile(packed,tile);
        for (unsigned i=0;i<256;++i) {
            expect(states[i] == oracle_state(packed,i), "16-bit state/window/wrap mismatch");
            expect(tile[perm[i]] == oracle_mcg(states[i]), "inner tile codebook mapping");
        }
    }
    std::uint16_t transitions[256];
    for (unsigned i=0;i<256;++i) transitions[i] = static_cast<std::uint16_t>((i*7+3)&15);
    pack_trellis_k4(transitions,packed);
    for (unsigned w=0;w<32;++w) {
        std::uint32_t expected=0;
        for (unsigned j=0;j<8;++j) expected=(expected<<4)|transitions[8*w+j];
        expect(packed[2*w] == (expected&65535) && packed[2*w+1] == (expected>>16), "packed byte order");
    }
    std::cout << "65536 MCG states, 2560 raw windows, permutation and pack layout passed\n";
}
