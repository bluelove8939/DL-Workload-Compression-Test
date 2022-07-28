#include "compression.h"


int main(int argc, char const *argv[]) {
    // Testbench for BDI algorithm
    CacheLine original = make_memory_chunk(CACHE128SIZ, 0);
    
    for (int offset = 0; offset < 4; offset++) {
        set_value(original.body, 0x803F0000, offset * 32 + 0,  4);
        set_value(original.body, 0x803F0004, offset * 32 + 4,  4);
        set_value(original.body, 0x803F0040, offset * 32 + 8,  4);
        set_value(original.body, 0x803F0044, offset * 32 + 12, 4);
        set_value(original.body, 0x803F0020, offset * 32 + 16, 4);
        set_value(original.body, 0x803F0014, offset * 32 + 20, 4);
        set_value(original.body, 0x803F0010, offset * 32 + 24, 4);
        set_value(original.body, 0x803F0004, offset * 32 + 28, 4);
    }

    CompressionResult result = bitplane_compression(original);
    print_compression_result(result);
    printf("\n");

    remove_compression_result(result);
    remove_memory_chunk(original);

    return 0;
}