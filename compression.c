#include "compression.h"


/* 
 * Functions for managing ByteArr and ValueBuffer 
 *   ByteArr is literally a structure for byte array representing memory block.
 *   ValueBuffer is 8Byte buffer for reading and writing value to ByteArr.
 * 
 * Functions:
 *   set_value: set value of the byte array
 *   set_value_bitwise: set value of the byte array with bitwise offset and size argument
 *   get_value: get value of the byte array
 *   get_value_bitwise: get value of the byte array with bitwise offset and size argument
 */

void set_value(ByteArr arr, ValueBuffer val, int offset, int size) {
    ValueBuffer mask = 1;
    for (int i = 0; i < size * BYTE_BITWIDTH; i++) {
        arr[(i / BYTE_BITWIDTH) + offset] |= ((((mask << i) & val) >> ((i / BYTE_BITWIDTH) * BYTE_BITWIDTH)));
    }
}

void set_value_bitwise(ByteArr arr, ValueBuffer val, int offset, int size) {
    ValueBuffer mask = 1;
    for (int i = 0; i < size; i++) {
        arr[(i + offset) / BYTE_BITWIDTH] |= ((val & (mask << i)) >> i) << ((i + offset) % BYTE_BITWIDTH);
    }
}

ValueBuffer get_value(ByteArr arr, int offset, int size) {
    ValueBuffer val = 0;
    for (int i = 0; i < size; i++) {
        val += ((ValueBuffer)arr[offset + i] << (i * BYTE_BITWIDTH));
    }
    if (size == 8) return val;
    return SIGNEX(val, size * BYTE_BITWIDTH - 1);
}

ValueBuffer get_value_bitwise(ByteArr arr, int offset, int size) {
    ValueBuffer val = 0;
    ValueBuffer mask = 1;
    int block_idx, block_bit_idx, output_bit_idx;
    for (int i = 0; i < size; i++) {
        block_idx = (i + offset) / BYTE_BITWIDTH;
        block_bit_idx = (i + offset) % BYTE_BITWIDTH;
        output_bit_idx = i;
        val += ((ValueBuffer)arr[block_idx] & (mask << (block_bit_idx))) >> block_bit_idx << output_bit_idx;
    }
    return val;
}


/* 
 * Functions for managing MemoryChunk
 *   MemoryChunk is a structure for representing memory blocks e.g. cache line and DRAM.
 *   Functions below are for managing those memory blocks.
 * 
 * Functions:
 *   make_memory_chunk: makes an array with its given size and bitwidth
 *   copy_memory_chunk: copy a given memory chunk
 *   remove_memory_chunk: removes memory chunk
 *   remove_compression_result: removes compression result (compressed memory chunk and overhead)
 *   print_memory_chunk: print content stored in the memory chunk in hexadecimal form
 *   print_memory_chunk_bitwise: print content stored in the memory chunk in binary form
 *   print_compression_result: prints compression result
 *   file2cacheline: convert binary file to cacheline
 */

MemoryChunk make_memory_chunk(int size, int initial) {
    MemoryChunk result;
    result.size = size;
    result.valid_bitwidth = size * BYTE_BITWIDTH;  // default valid bitwidth = size * 8
    result.body = (char *)malloc(size);
    memset(result.body, initial, size);
    return result;
}

MemoryChunk copy_memory_chunk(MemoryChunk target) {
    MemoryChunk chunk;
    chunk.size = target.size;
    chunk.valid_bitwidth = target.valid_bitwidth;
    chunk.body = (ByteArr)malloc(target.size);
    memcpy(chunk.body, target.body, target.size);
    return chunk;
}

void remove_memory_chunk(MemoryChunk chunk) {
    free(chunk.body);
}

void remove_compression_result(CompressionResult result) {
    remove_memory_chunk(result.compressed);
    remove_memory_chunk(result.overhead);
}

void print_memory_chunk(MemoryChunk chunk) {
    if (chunk.size % 4 != 0) {
        for (int i = 0; i < 4 - (chunk.size % 4); i++) {
            printf("--");
        }
    }

    for (int i = chunk.size-1; i >= 0; i--) {
        printf("%02x", chunk.body[i] & 0xff);
        if (i % 4 == 0) printf(" ");
    }
}

void print_memory_chunk_bitwise(MemoryChunk chunk) {
    Byte buffer, mask = 1;

    for (int i = chunk.valid_bitwidth-1; i >= 0; i--) {
        buffer = chunk.body[i / BYTE_BITWIDTH];
        printf("%d", (buffer & mask << (i % BYTE_BITWIDTH)) != 0);
    }
}

void print_compression_result(CompressionResult result) {
    Byte buffer, mask = 1;

    printf("======= Compression Results =======\n");
    printf("type: %s\n", result.compression_type);
    printf("original   size: %dBytes\n", result.original.size);
    printf("compressed size: %dBytes\n", result.compressed.size);
    printf("valid bitwidth:  %dbits\n", result.compressed.valid_bitwidth);
    printf("compression ratio: %.4f\n", (double)result.original.size / result.compressed.size);
    printf("original:   ");
    print_memory_chunk(result.original);
    printf("\n");
    printf("compressed: ");
    print_memory_chunk(result.compressed);
    printf("\n");
    printf("overhead: ");
    print_memory_chunk_bitwise(result.overhead);
    printf(" (%dbits)\n", result.overhead.valid_bitwidth);
    printf("is compressed: %s\n", result.is_compressed ? "true" : "false");
    printf("===================================\n");
}

MemoryChunk file2memorychunk(char const *filename, int offset, int size) {
    MemoryChunk chunk = make_memory_chunk(size, 0);
    char *buffer = (char *)malloc(size * sizeof(char));
    FILE *fp = fopen(filename, "rb");

    fseek(fp, offset, SEEK_SET);

    if (fp) {
        fread(buffer, size, sizeof(char), fp);
    } else {
        printf("opening file \'%s\' failed\n", filename);
    }

    memcpy(chunk.body, buffer, size);
    free(buffer);
    fclose(fp);

    return chunk;
}


/* 
 * Functions for BDI compression
 *   BDI(Base Delta Immediate) algorithm is a compression algorithm usually used to compress
 *   memory blocks e.g. cache line.
 * 
 * Functions:
 *   bdi_compression: BDI compression algorithm
 *   bdi_zero_packing: compress original cache line into 1Byte if all of the value is zero
 *   bdi_repeating: compress original cache line into 8Bytes if all of the value is repeating
 *   bdi_uni_base_packing: compress original cache line with given encoding
 * 
 * Note
 *   This algorithm is reference to the paper of PACT12 conference
 *   url: https://users.ece.cmu.edu/~omutlu/pub/bdi-compression_pact12.pdf
 */

CompressionResult bdi_compression(CacheLine original) {
    CompressionResult result;

    result = bdi_zero_packing(original);
    if (result.is_compressed) return result;

    result = bdi_repeating(original);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 8, 1);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 4, 1);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 2, 1);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 8, 2);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 4, 2);
    if (result.is_compressed) return result;

    result = bdi_uni_base_packing(original, 8, 4);
    if (result.is_compressed) return result;

    return result;
}

CompressionResult bdi_zero_packing(CacheLine original) {
    CompressionResult result;
    BufferArr arr = (BufferArr)malloc(original.size);
    Bool flag = TRUE;

    memcpy(arr, original.body, original.size);

    for (int idx = 0; idx < original.size / BUFFERSIZ; idx++)
        if (arr[idx] != 0)
            flag = FALSE;

    result.original = original;
    if (flag) {
        result.compression_type = "BDI (zero packing)";
        result.compressed = make_memory_chunk(1, 0);
        result.is_compressed = TRUE;
    } else {
        result.compression_type = "BDI (failed)";
        result.compressed = copy_memory_chunk(original);
        result.is_compressed = FALSE;
    }
    result.overhead = make_memory_chunk(2, 0);
    result.overhead.valid_bitwidth = 11;

    free(arr);

    return result;
}

CompressionResult bdi_repeating(CacheLine original) {
    CompressionResult result;
    BufferArr arr = (BufferArr)malloc(original.size);
    Bool flag = TRUE;

    memcpy(arr, original.body, original.size);

    for (int idx = 0; idx < original.size / BUFFERSIZ; idx++)
        if (arr[idx] != arr[0])
            flag = FALSE;

    result.original = original;
    if (flag) {
        result.compression_type = "BDI (repeating)";
        result.compressed = make_memory_chunk(BUFFERSIZ, 0);
        set_value(result.compressed.body, arr[0], 0, BUFFERSIZ);
        result.is_compressed = TRUE;
    } else {
        result.compression_type = "BDI (failed)";
        result.compressed = copy_memory_chunk(original);
        result.is_compressed = FALSE;
    }
    result.overhead = make_memory_chunk(2, 0);
    result.overhead.valid_bitwidth = 11;

    free(arr);

    return result;
}

CompressionResult bdi_uni_base_packing(CacheLine original, int k, int d) {
    CompressionResult result;
    ValueBuffer base, buffer, delta, bmask;

    result.compressed = make_memory_chunk(original.size, 0);
    base = get_value(original.body, 0, k);
    set_value(result.compressed.body, base, 0, k);
    result.compressed.size = k;

    for (int i = k; i < original.size; i += k) {
        buffer = get_value(original.body, i, k);
        delta = buffer - base;

        bmask = 0x00;
        if (d >= 1) bmask += 0xff;
        if (d >= 2) bmask += 0xff00;
        if (d >= 4) bmask += 0xffff0000;
        if (delta != SIGNEX(delta & bmask, (d * BYTE_BITWIDTH) - 1)) {
            result.original = original;
            result.compression_type = "BDI (failed)";
            remove_memory_chunk(result.compressed);
            result.compressed = copy_memory_chunk(original);
            result.is_compressed = FALSE;
            result.overhead = make_memory_chunk(2, 0);
            result.overhead.valid_bitwidth = 11;

            return result;
        }

        set_value(result.compressed.body, delta, result.compressed.size, d);
        result.compressed.size += d;
    }

    result.original = original;
    result.compression_type = "BDI (uni-base packing)";
    result.compressed.valid_bitwidth = result.compressed.size * BYTE_BITWIDTH;
    result.is_compressed = TRUE;
    result.overhead = make_memory_chunk(2, 0);
    result.overhead.valid_bitwidth = 11;

    return result;
}


/* 
 * Functions for BPC
 *   BPC(Bit-Plane Compression) algorithm
 * 
 * Functions:
 *   dbp_transformation: Delta + BitPlane transformation
 *   xor_transformation: XOR transformation
 *   dbx_transformation: Delta + BitPlane + XOR transformation (native)
 *   bitplane_compression: BPC algorithm
 * 
 * Note
 *   This algorithm is reference to the paper of 2016 ACM/IEEE
 *   Title: Bit-Plane Compression: Transforming Data for Better Compression in Many-core Architectures
 *   URL: https://ieeexplore.ieee.org/abstract/document/7551404
 */

CompressionResult dbp_transformation(CacheLine original, int dstep) {
    CompressionResult result;
    ValueBuffer base, buffer, mask = 1;
    CacheLine delta_arr = make_memory_chunk(original.size * 2, 0);
    CacheLine dbp_arr = make_memory_chunk(original.size * 2, 0);

    // Delta transformation
    base = get_value(original.body, 0, dstep);
    set_value(delta_arr.body, base, 0, dstep);
    delta_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = dstep; i < original.size; i += dstep) {
        buffer = get_value(original.body, i, dstep) - get_value(original.body, i - dstep, dstep);
        set_value_bitwise(delta_arr.body, buffer, delta_arr.valid_bitwidth, dstep * BYTE_BITWIDTH + 1);
        delta_arr.valid_bitwidth += dstep * BYTE_BITWIDTH + 1;
    }

    // BitPlane transformation
    set_value(dbp_arr.body, base, 0, dstep);
    dbp_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = 0; i < dstep * BYTE_BITWIDTH + 1; i++) {
        for (int j = 0; j < original.size / dstep - 1; j++) {
            buffer = (get_value_bitwise(delta_arr.body, dstep * BYTE_BITWIDTH + (dstep * BYTE_BITWIDTH + 1) * j, dstep * BYTE_BITWIDTH + 1) & (mask << i)) ? 1 : 0;
            set_value_bitwise(dbp_arr.body, buffer, dbp_arr.valid_bitwidth, 1);
            dbp_arr.valid_bitwidth += 1;
        }
    }

    remove_memory_chunk(delta_arr);

    result.original = original;
    result.compression_type = "DBP transformation";
    result.compressed = dbp_arr;
    result.compressed.size = ceil(dbp_arr.valid_bitwidth / BYTE_BITWIDTH);
    result.is_compressed = TRUE;
    result.overhead = make_memory_chunk(1, 0);
    result.overhead.valid_bitwidth = 0;

    return result;
}

CompressionResult xor_transformation(CacheLine dbp_arr, int dstep) {
    CompressionResult result;
    ValueBuffer base, buffer, mask = 1;
    CacheLine dbx_arr = make_memory_chunk(dbp_arr.size * 2, 0);

    // XOR transformation
    base = get_value(dbp_arr.body, 0, dstep);
    set_value(dbx_arr.body, base, 0, dstep);
    dbx_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = dstep * BYTE_BITWIDTH; i < dbp_arr.valid_bitwidth - (dstep * BYTE_BITWIDTH - 1); i += dstep * BYTE_BITWIDTH - 1) {
        buffer = get_value_bitwise(dbp_arr.body, i, dstep * BYTE_BITWIDTH - 1) ^ get_value_bitwise(dbp_arr.body, i + dstep * BYTE_BITWIDTH - 1, dstep * BYTE_BITWIDTH - 1);
        set_value_bitwise(dbx_arr.body, buffer, dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1);
        dbx_arr.valid_bitwidth += dstep * BYTE_BITWIDTH - 1;
    }

    set_value_bitwise(dbx_arr.body, get_value_bitwise(dbp_arr.body, dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1), dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1);
    dbx_arr.valid_bitwidth += dstep * BYTE_BITWIDTH - 1;

    result.original = dbp_arr;
    result.compression_type = "XOR transformation";
    result.compressed = dbx_arr;
    result.compressed.size = ceil(dbx_arr.valid_bitwidth / BYTE_BITWIDTH);
    result.is_compressed = TRUE;
    result.overhead = make_memory_chunk(1, 0);
    result.overhead.valid_bitwidth = 0;

    return result;
}

CompressionResult dbx_transformation(CacheLine original, int dstep) {
    CompressionResult result;
    ValueBuffer base, buffer, mask = 1;
    CacheLine delta_arr = make_memory_chunk(original.size * 2, 0);
    CacheLine dbp_arr = make_memory_chunk(original.size * 2, 0);
    CacheLine dbx_arr = make_memory_chunk(original.size * 2, 0);

    // Delta transformation
    base = get_value(original.body, 0, dstep);
    set_value(delta_arr.body, base, 0, dstep);
    delta_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = dstep; i < original.size; i += dstep) {
        buffer = get_value(original.body, i, dstep) - get_value(original.body, i - dstep, dstep);
        set_value_bitwise(delta_arr.body, buffer, delta_arr.valid_bitwidth, dstep * BYTE_BITWIDTH + 1);
        delta_arr.valid_bitwidth += dstep * BYTE_BITWIDTH + 1;
    }

    // BitPlane transformation
    set_value(dbp_arr.body, base, 0, dstep);
    dbp_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = 0; i < dstep * BYTE_BITWIDTH + 1; i++) {
        for (int j = 0; j < original.size / dstep - 1; j++) {
            buffer = (get_value_bitwise(delta_arr.body, dstep * BYTE_BITWIDTH + (dstep * BYTE_BITWIDTH + 1) * j, dstep * BYTE_BITWIDTH + 1) & (mask << i)) ? 1 : 0;
            set_value_bitwise(dbp_arr.body, buffer, dbp_arr.valid_bitwidth, 1);
            dbp_arr.valid_bitwidth += 1;
        }
    }

    // XOR transformation
    set_value(dbx_arr.body, base, 0, dstep);
    dbx_arr.valid_bitwidth = dstep * BYTE_BITWIDTH;

    for (int i = dstep * BYTE_BITWIDTH; i < dbp_arr.valid_bitwidth - (dstep * BYTE_BITWIDTH - 1); i += dstep * BYTE_BITWIDTH - 1) {
        buffer = get_value_bitwise(dbp_arr.body, i, dstep * BYTE_BITWIDTH - 1) ^ get_value_bitwise(dbp_arr.body, i + dstep * BYTE_BITWIDTH - 1, dstep * BYTE_BITWIDTH - 1);
        set_value_bitwise(dbx_arr.body, buffer, dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1);
        dbx_arr.valid_bitwidth += dstep * BYTE_BITWIDTH - 1;
    }

    set_value_bitwise(dbx_arr.body, get_value_bitwise(dbp_arr.body, dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1), dbx_arr.valid_bitwidth, dstep * BYTE_BITWIDTH - 1);
    dbx_arr.valid_bitwidth += dstep * BYTE_BITWIDTH - 1;

    remove_memory_chunk(delta_arr);
    remove_memory_chunk(dbp_arr);

    result.original = original;
    result.compression_type = "DBX transformation";
    result.compressed = dbx_arr;
    result.compressed.size = ceil(dbx_arr.valid_bitwidth / BYTE_BITWIDTH);
    result.is_compressed = TRUE;
    result.overhead = make_memory_chunk(1, 0);
    result.overhead.valid_bitwidth = 0;

    return result;
}

CompressionResult bitplane_compression(CacheLine original) {
    CompressionResult dbx_transformed, dbp_transformed, result;
    ValueBuffer buffer, dbp_buffer, encoded, run_length = 0, mask = 1;
    int cnt, pos, idx;
    Bool flag;
    
    int dstep = 4;
    if (original.size == CACHE32SIZ) dstep = 2;

    dbp_transformed = dbp_transformation(original, dstep);
    dbx_transformed = xor_transformation(dbp_transformed.compressed, dstep);

    result.compressed = make_memory_chunk(dbx_transformed.compressed.size, 0);
    set_value(result.compressed.body, get_value(dbx_transformed.compressed.body, 0, dstep), 0, dstep);  // save base value
    result.compressed.valid_bitwidth = dstep * BYTE_BITWIDTH;
    
    for (int i = dstep * BYTE_BITWIDTH; i < dbx_transformed.compressed.valid_bitwidth - (dstep * BYTE_BITWIDTH - 1); i += dstep * BYTE_BITWIDTH - 1) {
        buffer = get_value_bitwise(dbx_transformed.compressed.body, i, dstep * BYTE_BITWIDTH - 1);
        dbp_buffer = get_value_bitwise(dbp_transformed.compressed.body, i, dstep * BYTE_BITWIDTH - 1);

        // Run-length encoding
        if (buffer == 0) {
            run_length++;
            continue;
        } else if (run_length == 1) {
            set_value_bitwise(result.compressed.body, 1, result.compressed.valid_bitwidth, 3);
            result.compressed.valid_bitwidth += 3;
            run_length = 0;
            continue;
        } else {
            set_value_bitwise(result.compressed.body, 1, result.compressed.valid_bitwidth, 2);
            set_value_bitwise(result.compressed.body, run_length-2, result.compressed.valid_bitwidth + 2, 5);
            result.compressed.valid_bitwidth += 7;
            run_length = 0;
            continue;
        }

        // All 1’s
        if (SIGNEX(buffer, dstep * BYTE_BITWIDTH - 2) == -1) {
            result.compressed.valid_bitwidth += 5;
            continue;
        }

        // DBX!=0 & DBP=0
        if ((buffer != 0) && (dbp_buffer == 0)) {
            set_value_bitwise(result.compressed.body, 1, result.compressed.valid_bitwidth, 5);
            result.compressed.valid_bitwidth += 5;
            continue;
        }

        // Consecutive two 1's or Single 1
        cnt = 0;
        flag = TRUE;

        for (idx = 0; idx < dstep * BYTE_BITWIDTH - 1; idx++) {
            if (buffer & (mask << idx)) {
                pos = idx;
                break;
            }
        }

        pos = idx;

        for (; idx < dstep * BYTE_BITWIDTH - 1; idx++) {
            if (buffer & (mask << idx)) cnt++;
            else break;
        }

        for (; idx < dstep * BYTE_BITWIDTH - 1; idx++)
            if (buffer & (mask << idx)) 
                flag = FALSE;
        
        if (flag && (0 < cnt && cnt < 3)) {
            if (cnt == 2)
                set_value_bitwise(result.compressed.body, 2, result.compressed.valid_bitwidth, 5);
            else
                set_value_bitwise(result.compressed.body, 3, result.compressed.valid_bitwidth, 5);
            set_value_bitwise(result.compressed.body, pos, result.compressed.valid_bitwidth + 5, 5);
            result.compressed.valid_bitwidth += 10;
            continue;
        }

        // Uncompressed
        set_value_bitwise(result.compressed.body, 1, result.compressed.valid_bitwidth, 1);
        set_value_bitwise(result.compressed.body, buffer, result.compressed.valid_bitwidth + 1, dstep * BYTE_BITWIDTH - 1);
        result.compressed.valid_bitwidth += dstep * BYTE_BITWIDTH;
    }

    buffer = get_value_bitwise(dbx_transformed.compressed.body, dbx_transformed.compressed.valid_bitwidth - (dstep * BYTE_BITWIDTH - 1), dstep * BYTE_BITWIDTH - 1);
    set_value_bitwise(result.compressed.body, buffer, result.compressed.valid_bitwidth, dstep * BYTE_BITWIDTH - 1);
    result.compressed.valid_bitwidth += dstep * BYTE_BITWIDTH - 1;

    remove_compression_result(dbp_transformed);
    remove_compression_result(dbx_transformed);

    result.original = original;
    result.compression_type = "BPC";
    result.compressed.size = ceil(result.compressed.valid_bitwidth / BYTE_BITWIDTH);
    result.is_compressed = TRUE;
    result.overhead = make_memory_chunk(1, 0);
    result.overhead.valid_bitwidth = 0;

    return result;
}