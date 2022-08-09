import numpy as np
from typing import Iterable
from compression.binary_array import array2binary, binary2array, integer2binary, binary2integer, binary_shrinkable
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
    base = array2binary(arr[0], wordwidth)
    diffs = [integer2binary(d, wordwidth+1) for d in arr[1:] - arr[0:-1]]
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

    return encoded

def bitplane_decompression(binarr: str, wordwidth: int, chunksize: int) -> np.ndarray:
    dbxs = []
    dbps = []
    base = int(binarr[0:wordwidth], 2)
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

    original = np.array([base] + list(map(lambda x: binary2integer(x, wordwidth+1), [''.join(diff) for diff in zip(*dbps)])))
    for idx in range(1, len(original)):
        original[idx] += original[idx-1]

    return original


# Functions for BDI algorithm
#
# Description
#   BDI(Base-Delta Immediate) algorithm is a compression method
#
# References
#   - Base-Delta-Immediate Compression: Practical Data Compression for On-Chip Caches
#     https://ieeexplore.ieee.org/abstract/document/7842950
#
# Functions
#   bdi_compression: compression method
#   bdi_zero_pack: zero packing method
#   bdi_repeating_pack: packing repeating array
#   bdi_twobase_pack: zero base and one other base are used
#   bdi_decompression: decompression method

def bdi_compression(arr: np.ndarray, wordwidth: int) -> str:
    original = array2binary(arr, wordwidth)
    compressed = original

    for encoding, (packing_method, *args) in enumerate([
        (bdi_zero_pack,), (bdi_repeating_pack,),
        (bdi_twobase_pack, 8, 1), (bdi_twobase_pack, 8, 2), (bdi_twobase_pack, 8, 4),
        (bdi_twobase_pack, 4, 1), (bdi_twobase_pack, 4, 2), (bdi_twobase_pack, 2, 1),
    ]):
        buffer = packing_method(arr, wordwidth, encoding, *args)
        compressed = buffer if len(buffer) < len(compressed) else compressed

    return compressed

def bdi_zero_pack(arr: np.ndarray, wordwidth: int, encoding: int) -> str:
    for num in arr:
        if num != 0:
            return '1000' + array2binary(arr, wordwidth)
    return bin(encoding)[2:].zfill(4)

def bdi_repeating_pack(arr: np.ndarray, wordwidth: int, encoding: int) -> str:
    block_size = 64  # check every 1Byte
    binarr = array2binary(arr.flatten(), wordwidth)
    binarr_sliced = [binarr[i:i+block_size] for i in range(0, len(binarr), block_size)]

    for num in binarr_sliced:
        if num != binarr_sliced[0]:
            return '1000' + binarr

    return bin(encoding)[2:].zfill(4) + binarr_sliced[0]

def bdi_twobase_pack(arr: np.ndarray, wordwidth: int, encoding: int, k: int, d: int) -> str:
    compressed = ''
    zeromask = ''
    base = 0
    binarr = array2binary(arr, wordwidth)

    for i in range(0, len(binarr), k * 8):
        binnum = binarr[i:i+(k*8)]
        if binary_shrinkable(binnum, d * 8):
            compressed += binnum[len(binnum) - d * 8:]
            zeromask += '0'
            continue

        if base == 0:
            base = binary2integer(binnum, k * 8)

        buffer = binary2integer(binnum, k * 8)
        delta = integer2binary(buffer - base, k * 8 + 1)

        if binary_shrinkable(delta, d * 8):
            compressed += delta[len(delta) - d * 8:]
            zeromask += '1'
        else:
            return '1000' + binarr

    return bin(encoding)[2:].zfill(4) + zeromask + integer2binary(base, k * 8) + compressed

def bdi_decompression(binarr: str, wordwidth: int, chunksize: int) -> np.ndarray:
    encoding = int(binarr[:4], 2)

    if encoding == 0:
        return np.array([0] * int(chunksize / (wordwidth / 8)))
    elif encoding == 1:
        original = np.array([binary2integer(binarr[4:68], 64)] * int(chunksize / 8))
        binarr = array2binary(original, 64)
        return binary2array(binarr, wordwidth=wordwidth)
    elif encoding == 2:
        k, d = 8, 1
    elif encoding == 3:
        k, d = 8, 2
    elif encoding == 4:
        k, d = 8, 4
    elif encoding == 5:
        k, d = 4, 1
    elif encoding == 6:
        k, d = 4, 2
    elif encoding == 7:
        k, d = 2, 1
    else:
        return binary2array(binarr, wordwidth)

    pivot = 4
    zeromask = binarr[pivot:pivot+int(chunksize / k)]
    pivot += int(chunksize / k)
    base = binary2integer(binarr[pivot:pivot+k*8], wordwidth=k*8)
    pivot += k*8
    compressed = [binarr[pivot:][i:i+d*8] for i in range(0, len(binarr[pivot:]), d*8)]
    original = []

    for zm, delta in zip(zeromask, compressed):
        if zm == '1':
            original.append(base + binary2integer(delta, wordwidth=d*8))
        else:
            original.append(binary2integer(delta, wordwidth=d*8))

    binarr = array2binary(np.array(original), wordwidth=k*8)
    return binary2array(binarr, wordwidth=wordwidth)


if __name__ == '__main__':
    import os

    from binary_array import print_binary
    from compression.custom_streams import FileStream

    filepath = os.path.join(os.curdir, '..', 'extractions_quant_wfile', 'ResNet50_Imagenet_output', 'layer1_output0')

    comp_method = bdi_compression
    decomp_method = bdi_decompression

    stream = FileStream()
    stream.load_filepath(filepath=filepath, dtype=np.dtype('int8'))
    chunksize = 32
    arr, compressed, decompressed = None, None, None

    comp_total = 0
    orig_total = 0
    uncomp_cnt = 0

    while True:
        arr = stream.fetch(chunksize)
        if arr is None:
            break

        try:
            compressed = comp_method(arr, wordwidth=8)
            decompressed = decomp_method(compressed, wordwidth=8, chunksize=chunksize)
            pass
        except Exception as e:
            print("error occurred")
            print(f"Error: {e}")
            print(f"original: {arr}")
            input("Press any key to continue")
            continue

        if np.array_equal(compressed, decompressed):
            print('\rbitplane compression invalid')
            print_binary(array2binary(arr, wordwidth=8), swidth=8, startswith='original:     ', endswith='\n')
            print(f"compressed:   {compressed}")
            print_binary(array2binary(decompressed, wordwidth=8), swidth=8, startswith='decompressed: ', endswith='\n')
            input("Press any key to continue\n")
        else:
            orig_total += len(array2binary(arr, wordwidth=8))
            comp_total += len(compressed)
            uncomp_cnt += 1 if len(compressed) >= len(array2binary(arr, wordwidth=8)) else 0

            print(f"\rcursor: {stream.cursor}/{stream.fullsize()}\t"
                  f"compression ratio: {len(array2binary(arr, wordwidth=8)) / len(compressed):.4f}"
                  f"({orig_total / comp_total:.4f}) uncomp_cnt: {uncomp_cnt}", end='')
