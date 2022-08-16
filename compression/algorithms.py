import math
import numpy as np
from typing import Iterable
from compression.binary_array import array2binary, binary2array
from compression.binary_array import binary_xor, binary_and, binary_not, binary_or


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


if __name__ == '__main__':
    import os

    from binary_array import print_binary
    from compression.custom_streams import FileStream
    from models.tools.progressbar import progressbar

    parent_dirname = os.path.join(os.curdir, '..', 'extractions_activations')
    filepath = os.path.join(parent_dirname, 'AlexNet_Imagenet_output', 'ReLU_0_output0')

    comp_method = ebp_compression
    decomp_method = ebp_decompression
    dtype = np.dtype('float32')
    wordwidth = 32
    chunksize = 64

    stream = FileStream()
    stream.load_filepath(filepath=filepath, dtype=dtype)
    arr, compressed, decompressed = None, None, None

    comp_total = 0
    orig_total = 0
    uncomp_cnt = 0

    def compare_array(arr1: np.ndarray, arr2: np.ndarray, thres: float=1e-6) -> bool:
        return (np.sum(arr1 - arr2) ** 2) < thres

    while True:
        arr = stream.fetch(chunksize)
        if arr is None:
            break

        compressed = comp_method(arr, wordwidth=wordwidth)
        decompressed = decomp_method(compressed, wordwidth=wordwidth, chunksize=chunksize, dtype=dtype)

        if not compare_array(arr, decompressed, thres=1e-6):
            print('\nbitplane compression invalid')
            print(f"raw: [{' '.join(list(map(str, arr)))}]")
            print_binary(array2binary(arr, wordwidth=wordwidth),          swidth=wordwidth, startswith='original:     ', endswith='\n')
            print_binary(compressed,                                      swidth=wordwidth, startswith='compressed:   ', endswith='\n')
            print_binary(array2binary(decompressed, wordwidth=wordwidth), swidth=wordwidth, startswith='decompressed: ', endswith='\n')

            input("Press any key to continue\n")
        else:
            orig_total += len(array2binary(arr, wordwidth=wordwidth))
            comp_total += len(compressed)
            uncomp_cnt += 1 if len(compressed) >= len(array2binary(arr, wordwidth=wordwidth)) else 0

            print(f"\r{progressbar(status=stream.cursor, total=stream.fullsize(), scale=50)}  "
                  f"cursor: {stream.cursor}/{stream.fullsize()}\t"
                  f"compression ratio: {len(array2binary(arr, wordwidth=wordwidth)) / len(compressed):.4f}"
                  f"({orig_total / comp_total:.4f}) uncomp_cnt: {uncomp_cnt}", end='')
