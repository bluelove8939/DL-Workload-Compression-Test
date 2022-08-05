#ifndef __COMPRESSION_ALGO_HEADER
#define __COMPRESSION_ALGO_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <fcntl.h>
#include <math.h>

#define BYTE_BITWIDTH  8
#define BYTESIZ        1
#define HWORDSIZ       2
#define WORDSIZ        4
#define DWORDSIZ       8
#define BUFFERSIZ      8
#define CACHE32SIZ     32   // 32Bytes cacheline
#define CACHE64SIZ     64   // 64Bytes cacheline
#define CACHE128SIZ    128  // 128Bytes cacheline

// Macros for sign extension and bit masking (8Bytes buffer)
#define SIGNEX(v, sb)  ((v) | (((v) & (1 << (sb))) ? ~((1 << (sb))-1) : 0))
#define BITMASK(b)     (0x0000000000000001 << ((b) * (BYTE_BITWIDTH)))
#define BYTEMASK(b)    (0x00000000000000ff << ((b) * (BYTE_BITWIDTH)))

// Boolean expression
#ifndef BOOLEAN_EXPR
#define BOOLEAN_EXPR

#define FALSE 0
#define TRUE  1
typedef char  Bool;

#endif

// Structures for representing memory chunks(blocks)
typedef uint8_t    Byte;         // 1Byte (8bit)
typedef uint8_t *  ByteArr;      // Byte array
typedef int8_t     ByteBuffer;   // 1Byte  buffer
typedef int16_t    HwordBuffer;  // 2Bytes buffer
typedef int32_t    WordBuffer;   // 4Bytes buffer
typedef int64_t    ValueBuffer;  // 8Bytes buffer
typedef int64_t *  BufferArr;    // 8Byte buffer array

typedef struct {
    int size;            // byte size
    int valid_bitwidth;  // valid bitwidth
    ByteArr body;        // actual memory block
} MemoryChunk;

typedef MemoryChunk CacheLine;  // memory block representing cacheline
typedef MemoryChunk MetaData;   // memory block represinting compression overheads (e.g. tag overhead)

typedef struct {
    char *compression_type;  // name of compression algorithm
    CacheLine original;      // original cacheline (given)
    CacheLine compressed;    // compressed cacheline (induced)
    Bool is_compressed;      // flag identifying whether the given cacheline is compressed
    MetaData overhead;       // tag overhead (e.g. encoding type, segment pointer ...)
} CompressionResult;

// Functions for managing ByteArr and ValueBuffer
void set_value(ByteArr arr, ValueBuffer val, int offset, int size);
void set_value_bitwise(ByteArr arr, ValueBuffer val, int offset, int size);
ValueBuffer get_value(ByteArr arr, int offset, int size);
ValueBuffer get_value_bitwise(ByteArr arr, int offset, int size);

// Functions for managing MemoryChunk
MemoryChunk make_memory_chunk(int size, int initial);
MemoryChunk copy_memory_chunk(MemoryChunk target);
void remove_memory_chunk(MemoryChunk chunk);
void remove_compression_result(CompressionResult result);
void print_memory_chunk(MemoryChunk chunk);
void print_memory_chunk_bitwise(MemoryChunk chunk);
void print_compression_result(CompressionResult result);
MemoryChunk file2memorychunk(char const *filename, int offset, int size);

// BDI(Base Delta Immediate) compression algorithm
CompressionResult bdi_compression(CacheLine original);
CompressionResult bdi_zero_packing(CacheLine original);
CompressionResult bdi_repeating(CacheLine original);
CompressionResult bdi_uni_base_packing(CacheLine original, int k, int d);
CompressionResult bdi_multi_base_packing(CacheLine original, int k, int d);

// BPC algorithm
CompressionResult dbp_transformation(CacheLine original, int dstep);
CompressionResult xor_transformation(CacheLine original, int dstep);
CompressionResult dbx_transformation(CacheLine original, int dstep);
CompressionResult bitplane_compression(CacheLine original);

#endif