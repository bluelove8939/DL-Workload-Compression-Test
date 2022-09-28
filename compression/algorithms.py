import math
import zlib
import numpy as np
from typing import Iterable, Callable
from compression.binary_array import array2binary, binary2array
from compression.binary_array import binary_xor, binary_and, binary_not, binary_or
from compression.binary_array import trunc_array2binary, binary2integer, each_word_shrinkable


# Functions for BPC algorithm
#
# Description
#   BPC(Bit Plane Compression) algorithm is a compression method
#
# References
#   - Bit-Plane Compression: Transforming Data for Better Compression in Many-core Architectures, ISCA2016
#   - https://github.com/lukasc-ch/ExtendedBitPlaneCompression/tree/master/algoEvals
#
# Functions
#   delta_transform: method for delta transformation
#   dbp_transform: method for delta+bitplane transformation
#   dbx_transform: method for delta+bitplane+xor transformation
#   bitplane_compression: method for BPC algorithm

def delta_transform(arr: np.ndarray, wordwidth: int) -> Iterable:
    delta_width = wordwidth+1 if 'int' in arr.dtype.name else wordwidth
    base = array2binary(arr[0], wordwidth)
    diffs = [array2binary(d, delta_width) for d in arr[1:] - arr[0:-1]]
    return base, diffs

def dbp_transform(arr: np.ndarray, wordwidth: int) -> Iterable:
    base, diffs = delta_transform(arr, wordwidth)
    dbp = [''.join(bp) for bp in zip(*diffs)]
    return base, dbp

def dbx_transform(arr: np.ndarray, wordwidth: int) -> Iterable:
    base, dbp = dbp_transform(arr, wordwidth)
    dbx = [dbp[0]] + [binary_xor(a, b) for a, b in zip(dbp[:-1], dbp[1:])]
    return base, dbp, dbx

def bitplane_compression(arr: np.ndarray, wordwidth: int) -> str:
    base, dbps, dbxs = dbx_transform(arr, wordwidth)
    arrlen = arr.flatten().shape[0]
    encoded = base
    run_cnt = 0

    for dbp, dbx in zip(dbps, dbxs):
        if dbx == '0' * (len(arr) - 1):
            run_cnt += 1
            continue
        else:
            if run_cnt == 1:
                encoded += '001'
            elif run_cnt > 1:
                encoded += '01' + bin(run_cnt-2)[2:].zfill(int(np.ceil(np.log2(wordwidth+1))))
            run_cnt = 0

        if dbx == '1' * (len(arr) - 1):
            encoded += '00000'
        elif dbp == '0' * (len(arr) - 1):
            encoded += '00001'
        elif dbx.count('1') == 2 and dbx.count('11') == 1:
            encoded += '00010' + bin(dbx.find('11'))[2:].zfill(int(np.ceil(np.log2(arrlen))))
        elif dbx.count('1') == 1:
            encoded += '00011' + bin(dbx.find('1'))[2:].zfill(int(np.ceil(np.log2(arrlen))))
        else:
            encoded += '1' + dbx

    if run_cnt == 1:
        encoded += '001'
    elif run_cnt > 1:
        encoded += '01' + bin(run_cnt - 2)[2:].zfill(int(np.ceil(np.log2(wordwidth + 1))))

    return encoded

def bitplane_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    dbxs = []
    dbps = []
    # base = int(binarr[0:wordwidth], 2)
    base = binary2array(binarr[0:wordwidth], wordwidth=wordwidth, dtype=dtype)
    arrsize = int(chunksize / (wordwidth / 8))
    cursor = wordwidth

    while cursor < len(binarr):
        if binarr[cursor:cursor+2] == '01':
            cnt = int(binarr[cursor+2:cursor+2+int(np.ceil(np.log2(wordwidth+1)))], 2) + 2
            for _ in range(cnt):
                dbxs.append('0' * (arrsize - 1))
                dbps.append(dbps[-1] if len(dbps) != 0 else '0' * (arrsize - 1))
            cursor += 2 + int(np.ceil(np.log2(wordwidth+1)))
        elif binarr[cursor:cursor+3] == '001':
            dbxs.append('0' * (arrsize - 1))
            dbps.append(dbps[-1] if len(dbps) != 0 else '0' * (arrsize - 1))
            cursor += 3
        elif binarr[cursor:cursor+5] == '00000':
            dbxs.append('1' * (arrsize - 1))
            if len(dbxs) == 1:
                dbps.append('1' * (arrsize - 1))
            else:
                dbps.append(binary_not(dbps[-1]))
            cursor += 5
        elif binarr[cursor:cursor+5] == '00001':
            dbps.append('0' * (arrsize - 1))
            dbxs.append(binary_xor(dbps[-2], dbps[-1]))
            cursor += 5
        elif binarr[cursor:cursor+5] == '00010':
            offset = int(binarr[cursor+5:cursor+5+int(np.ceil(np.log2(arrsize)))], 2)
            dbxs.append('0' * offset + '11' + '0' * (arrsize - offset - 3))
            if len(dbps) != 0:
                dbps.append(binary_or(binary_and(binary_not(dbps[-1]), dbxs[-1]), binary_and(dbps[-1], binary_not(dbxs[-1]))))
            else:
                dbps.append(dbxs[-1])
            cursor += 5 + int(np.ceil(np.log2(arrsize)))
        elif binarr[cursor:cursor+5] == '00011':
            offset = int(binarr[cursor+5:cursor+5+int(np.ceil(np.log2(arrsize)))], 2)
            dbxs.append('0' * offset + '1' + '0' * (arrsize - offset - 2))
            if len(dbps) != 0:
                dbps.append(binary_or(binary_and(binary_not(dbps[-1]), dbxs[-1]), binary_and(dbps[-1], binary_not(dbxs[-1]))))
            else:
                dbps.append(dbxs[-1])
            cursor += 5 + int(np.ceil(np.log2(arrsize)))
        elif binarr[cursor] == '1':
            dbxs.append(binarr[cursor+1:cursor+arrsize])
            if len(dbps) != 0:
                dbps.append(binary_or(binary_and(binary_not(dbps[-1]), dbxs[-1]), binary_and(dbps[-1], binary_not(dbxs[-1]))))
            else:
                dbps.append(dbxs[-1])
            cursor += arrsize

    delta_dtype = np.dtype('int32') if 'int' in dtype.name else np.dtype('float32')
    delta_width  = wordwidth+1 if 'int' in dtype.name else wordwidth

    original = np.array(list(base) + list(binary2array(''.join([''.join(diff) for diff in zip(*dbps)]), delta_width, dtype=delta_dtype)), dtype=dtype)
    for idx in range(1, len(original)):
        original[idx] += original[idx-1]

    return original.astype(dtype=dtype)


# Functions for Zero-RLE algorithm
#
# Description
#   Zero-RLE(Zero Run Length Encoding) algorithm is a compression method
#
# Functions
#   zrle_compression
#   zrle_decompression

def zrle_compression(arr: np.ndarray, wordwidth: int, max_burst_len=16,) -> str:
    encoded = ''
    run_length = 0

    for idx, num in enumerate(arr.flatten()):
        if num == 0:
            run_length += 1

        if run_length == max_burst_len or (num != 0 and run_length > 0):
            encoded += '0' + bin(run_length-1)[2:].zfill(math.ceil(np.log2(max_burst_len)))
            run_length = 0

        if num != 0:
            encoded += '1' + array2binary(num, wordwidth)

    if run_length > 0:
        encoded += '0' + bin(run_length-1)[2:].zfill(math.ceil(np.log2(max_burst_len)))

    return encoded

def zrle_decompression(binarr: str, wordwidth: int, chunksize: int, max_burst_len=16, dtype=np.dtype('int8')) -> np.ndarray:
    arr = []
    cntwidth = math.ceil(np.log2(max_burst_len))
    cursor = 0

    while cursor < len(binarr):
        if binarr[cursor] == '0':
            cnt = int(binarr[cursor+1:cursor+1+cntwidth], 2) + 1
            arr += [0] * cnt
            cursor += 1 + cntwidth
        else:
            arr += list(binary2array(binarr[cursor+1:cursor+1+wordwidth], wordwidth, dtype=dtype))
            cursor += 1 + wordwidth

    return np.array(arr, dtype=dtype)


# Functions for ZVC algorithm
#
# Description
#   ZVC(Zero Value Compression) algorithm is a compression method
#
# Functions
#   zeroval_compression
#   zeroval_decompression

def zeroval_compression(arr: np.ndarray, wordwidth: int,) -> str:
    zerovec = ''.join(['0' if val == 0 else '1' for val in arr])
    nonzerovec = ''.join(['' if val == 0 else array2binary(val, wordwidth=wordwidth) for val in arr])
    return zerovec + nonzerovec

def zeroval_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    arr = []
    cursor = 0
    zerovec_length = int(chunksize / (wordwidth / 8))

    zerovec = binarr[0:zerovec_length]
    nonzerovec = binarr[zerovec_length:]

    for zbit in zerovec:
        if zbit == '0':
            arr.append(0)
        else:
            arr += list(binary2array(nonzerovec[cursor:cursor+wordwidth], wordwidth, dtype=dtype))
            cursor += wordwidth

    return np.array(arr, dtype=dtype)


# Functions for EBPC algorithm
#
# Description
#   EBPC(Extended Bit-Plane Compression) algorithm is a compression method
#
# Functions
#   ebp_compression
#   ebp_decompression

def ebp_compression(arr: np.ndarray, wordwidth: int, max_burst_len=16,) -> str:
    zrle_encoded = zrle_compression(arr, wordwidth=wordwidth, max_burst_len=max_burst_len)
    bpc_encoded = bitplane_compression(arr, wordwidth=wordwidth)

    if len(zrle_encoded) <= len(bpc_encoded):
        return '0' + zrle_encoded
    return '1' + bpc_encoded

def ebp_decompression(binarr: str, wordwidth: int, chunksize: int, max_burst_len=16, dtype=np.dtype('int8')) -> np.ndarray:
    if binarr[0] == '0':
        decoded = zrle_decompression(binarr[1:], wordwidth=wordwidth, chunksize=chunksize,
                                     max_burst_len=max_burst_len, dtype=dtype)
    else:
        decoded = bitplane_decompression(binarr[1:], wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)
    return decoded


# Functions for zlib (dummy for official Zlib library)
#
# Functions
#   zlib_compression
#   zlib_decompression

def zlib_compression(arr: np.ndarray, wordwidth: int,) -> str:
    barr = zlib.compress(arr.tobytes())
    iarr = np.frombuffer(barr, dtype=np.dtype('int8'))
    return array2binary(iarr, wordwidth=8)

def zlib_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    iarr = binary2array(binarr, wordwidth=8, dtype=np.dtype('int8'))
    barr = iarr.tobytes()
    return np.frombuffer(zlib.decompress(barr), dtype=dtype)


# Functions for BDI algorithm
#
# Description
#   Testing
#
# Functions
#   bdi_compression
#   bdi_decompression

def bdi1b_compression(arr: np.ndarray, wordwidth: int,) -> str:
    dtype = arr.dtype
    delta_width = wordwidth
    if 'int' in dtype.name:
        delta_width = wordwidth+1

    base = arr[0]
    deltas = arr[1:] - base
    validwidth, delta_binarr = trunc_array2binary(deltas, wordwidth=delta_width)
    return array2binary(base, wordwidth) + bin(validwidth)[2:].rjust(math.ceil(np.log2(delta_width)) + 1, '0') + delta_binarr

def bdi2b_compression(arr: np.ndarray, wordwidth: int,) -> str:
    dtype = arr.dtype
    delta_width = wordwidth
    if 'int' in dtype.name:
        delta_width = wordwidth + 1

    nonzero_idx = np.nonzero(arr)[0]

    if len(nonzero_idx) == 0:
        return '0000'

    nz_base = arr[nonzero_idx][0]
    widthunit = wordwidth // 8

    for encoding, validwidth in enumerate(range(widthunit, wordwidth, widthunit)):
        zero_base_bv = each_word_shrinkable(arr, wordwidth, validwidth)  # bool arr that checks whether each word is shrinkable with zero base

        deltas = arr - nz_base  # delta array with non-zero base
        nz_base_bv = each_word_shrinkable(deltas, delta_width, validwidth)  # bool arr that checks whether each word is shrinkable with zero base

        if np.count_nonzero(zero_base_bv + nz_base_bv) == len(arr):
            encoding_binnum = bin(encoding + 1)[2:].rjust(4, '0')
            base_bitmask = ''.join(['1' if val == False else '0' for val in zero_base_bv])
            encoded = ''

            bin_zb_raw = array2binary(arr, wordwidth=wordwidth)
            bin_nb_raw = array2binary(deltas, wordwidth=delta_width)

            binarr_zb = [bin_zb_raw[idx:idx+wordwidth] for idx in range(0, len(bin_zb_raw), wordwidth)]
            binarr_nb = [bin_nb_raw[idx:idx+delta_width] for idx in range(0, len(bin_nb_raw), delta_width)]

            for bm, zb, nb in zip(base_bitmask, binarr_zb, binarr_nb):
                if bm == '0':
                    encoded += zb[wordwidth-validwidth:]
                else:
                    encoded += nb[delta_width-validwidth:]

            return encoding_binnum + base_bitmask + array2binary(nz_base, wordwidth) + encoded

        return '1111' + array2binary(arr, wordwidth)


def bdizv_compression(arr: np.ndarray, wordwidth: int) -> str:
    zeromask = ''.join(['0' if val == 0 else '1' for val in arr])
    nonzero_arr = arr[np.nonzero(arr)]

    dtype = arr.dtype
    delta_width = wordwidth
    if 'int' in dtype.name:
        delta_width = wordwidth + 1

    if len(nonzero_arr) == 0:
        return zeromask
    if len(nonzero_arr) == 1:
        return zeromask + bin(8)[2:].rjust(math.ceil(np.log2(delta_width)) + 1,'0') + array2binary(nonzero_arr[0], wordwidth)

    base = nonzero_arr[0]
    deltas = nonzero_arr[1:] - base
    validwidth, delta_binarr = trunc_array2binary(deltas, wordwidth=delta_width)

    return zeromask + bin(validwidth)[2:].rjust(math.ceil(np.log2(delta_width)) + 1,'0') + array2binary(base, wordwidth) + delta_binarr

def bdi1b_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    delta_width = wordwidth
    if 'int' in dtype.name:
        delta_width = wordwidth + 1

    width_bit = math.ceil(np.log2(delta_width)) + 1
    base = binarr[:wordwidth]
    validwidth = int(binarr[wordwidth:wordwidth+width_bit], 2)
    delta_binarr = binarr[wordwidth+width_bit:]

    # print(len(binarr), width_bit, wordwidth, validwidth,)

    if 'int' in dtype.name:
        base = binary2integer(base, wordwidth=wordwidth)
        original = [base]
        for idx in range(0, len(delta_binarr), validwidth):
            binnum = delta_binarr[idx:idx+validwidth]
            original.append(binary2integer(binnum, wordwidth=validwidth) + base)
    else:
        base = binary2array(base, wordwidth=wordwidth, dtype=dtype)
        deltas = list(binary2array(delta_binarr, wordwidth=validwidth, dtype=dtype))
        original = list(base) + list(deltas + base)

    return np.array(original, dtype=dtype)

def bdi2b_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    arrlen = int(chunksize / wordwidth * 8)
    encoding = binarr[:4]

    if encoding == '0000':
        return np.array([0] * arrlen, dtype=dtype)
    if encoding == '1111':
        return binary2array(binarr[4:], wordwidth, dtype)

    base_bitmask = binarr[4:4+arrlen]
    nz_base = binary2integer(binarr[4+arrlen:4+arrlen+wordwidth], wordwidth=wordwidth) \
        if 'int' in dtype.name \
        else binary2array(binarr[4+arrlen:4+arrlen+wordwidth], wordwidth=wordwidth, dtype=dtype)
    validwidth = int(encoding, 2)
    encoded = binarr[4+arrlen+wordwidth:]
    arr = []

    for idx in range(0, len(encoded), validwidth):
        if 'int' in dtype.name:
            num = binary2integer(encoded[idx:idx + validwidth], wordwidth=validwidth)
        else:
            num = binary2array(encoded[idx:idx+validwidth], wordwidth=validwidth, dtype=dtype)
        arr.append(num if base_bitmask[idx] == '0' else num + nz_base)

    return np.array(arr, dtype=dtype)

def bdizv_decompression(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
    delta_width = wordwidth
    if 'int' in dtype.name:
        delta_width = wordwidth + 1

    arrlen = int(chunksize / wordwidth * 8)
    nonzero_arrlen = binarr[:arrlen].count('1')
    zeromask = np.array([ch == '1' for ch in binarr[:arrlen]], dtype=bool)
    width_bit = math.ceil(np.log2(delta_width)) + 1

    if nonzero_arrlen == 0:
        return np.zeros(shape=arrlen, dtype=dtype)

    validwidth = int(binarr[arrlen:arrlen + width_bit], 2)

    base = binary2integer(binarr[arrlen + width_bit:arrlen + width_bit + wordwidth], wordwidth=wordwidth) \
        if 'int' in dtype.name \
        else binary2array(binarr[arrlen + width_bit:arrlen + width_bit + wordwidth], wordwidth=wordwidth, dtype=dtype)

    if nonzero_arrlen == 1:
        original = np.zeros(shape=arrlen, dtype=dtype)
        original[zeromask] = base
        return original

    delta_binarr = binarr[arrlen + wordwidth + width_bit:]

    if 'int' in dtype.name:
        nonzero_arr = [base]
        for idx in range(0, len(delta_binarr), validwidth):
            binnum = delta_binarr[idx:idx+validwidth]
            nonzero_arr.append(binary2integer(binnum, wordwidth=validwidth) + base)
    else:
        deltas = list(binary2array(delta_binarr, wordwidth=validwidth, dtype=dtype))
        nonzero_arr = list(base) + list(deltas + base)

    nonzero_arr = np.array(nonzero_arr, dtype=dtype)
    original = np.zeros(shape=arrlen, dtype=dtype)
    original[zeromask] = nonzero_arr

    return original

bdi_compression: Callable   = bdizv_compression
bdi_decompression: Callable = bdizv_decompression


# Functions for EBDI algorithm
#
# Description
#   Test
#
# Functions
#   ebdi_compression
#   ebdi_decompression

def ebdi_compression(arr: np.ndarray, wordwidth: int, max_burst_len=16,) -> str:
    zrle_encoded = zrle_compression(arr, wordwidth=wordwidth, max_burst_len=max_burst_len)
    bdi_encoded = bdi_compression(arr, wordwidth=wordwidth)

    if len(zrle_encoded) <= len(bdi_encoded):
        return '0' + zrle_encoded
    return '1' + bdi_encoded

def ebdi_decompression(binarr: str, wordwidth: int, chunksize: int, max_burst_len=16, dtype=np.dtype('int8')) -> np.ndarray:
    if binarr[0] == '0':
        decoded = zrle_decompression(binarr[1:], wordwidth=wordwidth, chunksize=chunksize, max_burst_len=max_burst_len, dtype=dtype)
    else:
        decoded = bdi_decompression(binarr[1:], wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)
    return decoded


# Functions for user custom complementary algorithms
#
# Description
#   Test
#
# Functions
#   complementary_compression
#   complementary_decompression

def complementary_compression(algo0, algo1) -> Callable:
    def compression_func(arr: np.ndarray, wordwidth: int,) -> str:
        encoded0 = algo0(arr, wordwidth=wordwidth)
        encoded1 = algo1(arr, wordwidth=wordwidth)

        if len(encoded0) <= len(encoded1):
            return '0' + encoded0
        return '1' + encoded1

    return compression_func

def complementary_decompression(algo0, algo1) -> Callable:
    def decompression_func(binarr: str, wordwidth: int, chunksize: int, dtype=np.dtype('int8')) -> np.ndarray:
        if binarr[0] == '0':
            decoded = algo0(binarr[1:], wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)
        else:
            decoded = algo1(binarr[1:], wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)
        return decoded

    return decompression_func



# Functions for CSC/CSR format
#
# Note
#   These functions are only for accelerator simulation
#   Returns simple concatenation of non-zero words, index vector and column pointers

from scipy.sparse import csr_matrix, csc_matrix

def csr_compression(arr: np.ndarray, wordwidth: int) -> str:
    compr_mat = csr_matrix(arr)

    nonzero_num = compr_mat.data.shape[0]
    index_width = math.ceil(math.log2(nonzero_num)) if nonzero_num != 0 else 1

    compr_data = array2binary(compr_mat.data, wordwidth=wordwidth)
    compr_indptr = array2binary(compr_mat.indptr, wordwidth=index_width)
    compr_indices = array2binary(compr_mat.indices, wordwidth=index_width)

    return compr_data + compr_indices + compr_indptr

def csc_compression(arr: np.ndarray, wordwidth: int) -> str:
    compr_mat = csc_matrix(arr)

    nonzero_num = compr_mat.data.shape[0]
    index_width = math.ceil(math.log2(nonzero_num)) if nonzero_num != 0 else 1

    compr_data = array2binary(compr_mat.data, wordwidth=wordwidth)
    compr_indptr = array2binary(compr_mat.indptr, wordwidth=index_width)
    compr_indices = array2binary(compr_mat.indices, wordwidth=index_width)

    return compr_data + compr_indices + compr_indptr


if __name__ == '__main__':
    import os

    from binary_array import print_binary
    from compression.custom_streams import FileStream
    from models.tools.progressbar import progressbar

    # parent_dirname = os.path.join(os.curdir, '..', 'extractions_quant_wfile')
    parent_dirname = os.path.join(os.curdir, '..', 'extractions_activations')
    # parent_dirname = os.path.join(os.curdir, '..', 'extractions_quant_activations')

    # filepath = os.path.join(parent_dirname, 'InceptionV3_Imagenet_output', 'ConvReLU2d_0_output2')
    # filepath = os.path.join(parent_dirname, 'AlexNet_Imagenet_output', 'ReLU_0_output0')
    filepath = os.path.join(parent_dirname, 'ResNet50_Imagenet_output', 'Conv2d_0_output0')

    comp_method = bdi1b_compression
    decomp_method = bdi1b_decompression
    dtype = np.dtype('float32')
    wordwidth = 32
    chunksize = 128
    vstep = 1000

    print('Algorithm Test Configs')
    print(f"- file:  {filepath}")
    print(f"- dtype: {dtype.name}")
    print(f"- chunksize: {chunksize}Byte")
    print(f"- wordwidth: {wordwidth}bit")
    print(f"- compression:   {comp_method.__name__}")
    print(f"- decompression: {decomp_method.__name__}\n")

    stream = FileStream()
    stream.load_filepath(filepath=filepath, dtype=dtype)
    arr, compressed, decompressed = None, None, None

    comp_total = 0
    orig_total = 0
    uncomp_cnt = 0
    iter_cnt = 0

    def compare_array(arr1: np.ndarray, arr2: np.ndarray, thres: float=1e-6) -> bool:
        return (np.sum(arr1 - arr2) ** 2) < thres

    while True:
        arr = stream.fetch(chunksize)
        if arr is None:
            break

        compressed = comp_method(arr, wordwidth=wordwidth)
        decompressed = decomp_method(compressed, wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)

        if not compare_array(arr, decompressed, thres=1e-8):
            print(f'\nbitplane compression invalid at {iter_cnt}')
            print(f"raw: [{' '.join(list(map(lambda x: f'{x:10.6f}' if 'float' in dtype.name else f'{x:3d}', arr)))}]")
            print(f"dec: [{' '.join(list(map(lambda x: f'{x:10.6f}' if 'float' in dtype.name else f'{x:3d}', decompressed)))}]")
            print_binary(array2binary(arr, wordwidth=wordwidth),          swidth=wordwidth, startswith='original:     ', endswith='\n')
            print_binary(compressed,                                      swidth=wordwidth, startswith='compressed:   ', endswith='\n')
            print_binary(array2binary(decompressed, wordwidth=wordwidth), swidth=wordwidth, startswith='decompressed: ', endswith='\n')

            input("Press any key to continue\n")
        else:
            orig_total += len(array2binary(arr, wordwidth=wordwidth))
            comp_total += len(compressed)
            uncomp_cnt += 1 if len(compressed) >= len(array2binary(arr, wordwidth=wordwidth)) else 0

            if iter_cnt % vstep == 0:
                print(f"\r{progressbar(status=stream.cursor, total=stream.fullsize(), scale=50)}  "
                      f"cursor: {stream.cursor}/{stream.fullsize()}\t"
                      f"compression ratio: {len(array2binary(arr, wordwidth=wordwidth)) / len(compressed):.4f}"
                      f"({orig_total / comp_total:.4f}) uncomp_cnt: {uncomp_cnt}", end='')

        iter_cnt += 1
