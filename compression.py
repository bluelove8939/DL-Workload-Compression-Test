from typing import Iterable, Callable
import numpy as np

from custom_streams import CustomStream


# Functions for managing value array and binary representation
#
# References
#   - Bit-Plane Compression: Transforming Data for Better Compression in Many-core Architectures, ISCA2016
#   - https://github.com/lukasc-ch/ExtendedBitPlaneCompression/tree/master/algoEvals
#
# Functions:
#   binary_xor: do exclusive OR operation by using given data with binary representation
#   array2binary: convert a numpy array to binary representation
#   print_binary: print binary represetation

def binary_shrinkable(num: str, d: int) -> bool:
    for i in range(len(num)-d):
        if num[0] != num[i]:
            return False
    return True

def binary_xor(a: str, b: str) -> str:
    wordwidth = len(a)
    return bin(int(a, 2) ^ int(b, 2))[2:].zfill(wordwidth)

def binary_not(a: str) -> str:
    return ''.join(['1' if letter == '0' else '0' for letter in a])

def integer2binary(num: int, wordwidth: int):  # 2's complement number conversion
    ret = ''
    for i in reversed(range(wordwidth)):
        ret += '1' if num & (1 << i) else '0'
    return ret

def binary2integer(num: str, wordwidth: int):  # 2's complement number conversion
    factor = 1
    if num[0] == '1':
        num = binary_not(integer2binary(int(num, 2) - 1, wordwidth))
        factor = -1
    return factor * int(num, 2)

def array2binary(arr: np.ndarray, wordwidth: int=None) -> str:
    barr = arr.byteswap().tobytes()
    precision = arr.dtype.itemsize * 8
    if wordwidth is None:
        wordwidth = precision
    rawbinarr = bin(int.from_bytes(barr, byteorder='big'))[2:].zfill(len(barr) * 8)
    binarr = ''.join([rawbinarr[i:i + precision][-wordwidth:] # if wordwidth <= precision
                      # else rawbinarr[i:i + precision].rjust(wordwidth, rawbinarr[i:i + precision][0])
                      for i in range(0, len(rawbinarr), precision)])
    return binarr

def binary2array(binarr: str, wordwidth: int) -> np.ndarray:
    arr = np.array([binary2integer(binarr[i:i+wordwidth], wordwidth) for i in range(0, len(binarr), wordwidth)])
    return arr

def print_binary(binstr: str, swidth: int=8, startswith='', endswith='\n') -> None:
    print(startswith, end='')
    for i in range(0, len(binstr), swidth):
        print(binstr[i:i+swidth], end=' ')
    print(endswith, end='')


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
    diffs = [array2binary(d, wordwidth+1) for d in arr[1:] - arr[0:-1]]
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
    chunksize = arr.flatten().shape[0]
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
                encoded += '01' + bin(run_cnt)[2:].zfill(5)
            run_cnt = 0

        if dbx == '1' * (len(arr) - 1):
            encoded += '00000'
        elif dbp == '0' * (len(arr) - 1):
            encoded += '00001'
        elif dbx.count('1') == 2 and dbx.count('11') == 1:
            encoded += '00010' + bin(dbx.find('11'))[2:].zfill(int(np.log2(chunksize)))
        elif dbx.count('1') == 1:
            encoded += '00011' + bin(dbx.find('1'))[2:].zfill(int(np.log2(chunksize)))
        else:
            encoded += '1' + dbx

    return encoded


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

def bdi_compression(arr: np.ndarray, wordwidth: int) -> str:
    original = array2binary(arr, wordwidth)
    compressed = original

    buffer = bdi_zero_pack(arr, wordwidth)
    compressed = buffer if len(buffer) < len(compressed) else compressed

    buffer = bdi_repeating_pack(arr, wordwidth)
    compressed = buffer if len(buffer) < len(compressed) else compressed

    for encoding, (k, d) in enumerate([(8, 1), (8, 2), (8, 4), (4, 1), (4, 2), (2, 1),]):
        buffer = bdi_twobase_pack(arr, wordwidth, k, d, encoding)
        compressed = buffer if len(buffer) < len(compressed) else compressed

    return compressed

def bdi_zero_pack(arr: np.ndarray, wordwidth: int) -> str:
    for num in arr:
        if num != 0:
            return array2binary(arr, wordwidth)
    return '0000' + '0' * 8

def bdi_repeating_pack(arr: np.ndarray, wordwidth: int) -> str:
    block_size = 64  # check every 1Byte
    binarr = array2binary(arr.flatten(), wordwidth)
    binarr_sliced = [binarr[i:i+block_size] for i in range(0, len(binarr), block_size)]

    for num in binarr_sliced:
        if num != binarr_sliced[0]:
            return binarr

    return '0001' + binarr_sliced[0]

def bdi_twobase_pack(arr: np.ndarray, wordwidth: int, k: int, d: int, encoding: int) -> str:
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
            return binarr

    return bin(encoding).zfill(4) + zeromask + integer2binary(base, k * 8) + compressed

def bdi_decompression(binarr: str, wordwidth: int, chunksize: int) -> np.ndarray:
    encoding = int(binarr[:4], 2)

    if encoding == 0:
        return np.array([0] * int(chunksize / (wordwidth / 8)))
    elif encoding == 1:
        return np.array([binary2integer(binarr[4:68], wordwidth)] * int(chunksize / (wordwidth / 8)))
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
    zeromask = binarr[pivot:pivot + int(np.log2(chunksize / k))]
    pivot += int(np.log2(chunksize / k))
    base = binarr[pivot:pivot+k*8]
    pivot += k*8
    compressed = binarr[pivot:]

    return np.array([])


# Compressor module
#
# Description
#   Module for simulating compression of data stream (from file or rawdata)
#   Data is fetched from the specific module 'CustomStream'
#
# Methods:
#   step: compress data only one step (fetch data with the given bandwidth from the source)
#   calc_compression_ratio: calculate compression ratio

class Compressor(object):
    STOPCODE = 'STOPCODE'

    def __init__(self, cmethod: Callable, stream: CustomStream, bandwidth: int=128, wordbitwidth: int=32):
        self.cmethod = cmethod            # compression method (needs to follow the given interface)
        self.bandwidth = bandwidth        # bandwidth in Bytes
        self.wordbitwidth = wordbitwidth  # wordwidth in bits
        self.stream = stream              # to fetch chunk from data or file

    def step(self, verbose: int=1) -> str:
        stepsiz = int(self.bandwidth)
        arr = self.stream.fetch(size=stepsiz)

        if arr is None:
            return Compressor.STOPCODE

        binarr = self.cmethod(arr, self.wordbitwidth)

        if verbose == 2:
            print_binary(array2binary(arr, self.wordbitwidth), self.wordbitwidth, startswith='original:   ', endswith='\n')
            print_binary(binarr, self.wordbitwidth, startswith='compressed: ', endswith='\n')

        return binarr

    def calc_compression_ratio(self, maxiter: int=-1, verbose: int=1) -> float:
        total_original_size = 0
        total_compressed_size = 0
        cntiter = 0

        self.stream.reset()

        while True:
            binarr = self.step(verbose)
            if binarr == Compressor.STOPCODE:
                return total_original_size / total_compressed_size if total_compressed_size != 0 else 0

            original_size = self.bandwidth * 8
            compressed_size = len(binarr)

            total_original_size += original_size
            total_compressed_size += compressed_size

            if verbose == 1:
                print(f"\rcursor: {self.stream.cursor}/{self.stream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {original_size / compressed_size:.6f} "
                      f"({total_original_size / total_compressed_size:.6f})", end='          ')
            elif verbose == 2:
                print(f"\rcursor: {self.stream.cursor}/{self.stream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {self.bandwidth * 8 / len(binarr):.6f} "
                      f"({total_original_size / total_compressed_size:.6f})")

            cntiter += 1
            if maxiter != -1 and cntiter >= maxiter:
                break

        return original_size / compressed_size


class BitPlaneCompressor(Compressor):
    def __init__(self, stream: CustomStream, bandwidth: int=128, wordbitwidth: int=32):
        super(BitPlaneCompressor, self).__init__(cmethod=bitplane_compression,
                                                 stream=stream,
                                                 bandwidth=bandwidth,
                                                 wordbitwidth=wordbitwidth)

class BDICompressor(Compressor):
    def __init__(self, stream: CustomStream, bandwidth: int = 128, wordbitwidth: int = 32):
        super(BDICompressor, self).__init__(cmethod=bdi_compression,
                                            stream=stream,
                                            bandwidth=bandwidth,
                                            wordbitwidth=wordbitwidth)


if __name__ == '__main__':
    from custom_streams import DataStream

    # rawdata = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,], dtype=np.dtype(int))

    rawdata = np.array([1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1, 5,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ], dtype=np.dtype('int8'))

    # rawdata = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], dtype=np.dtype('int8'))

    stream = DataStream()
    stream.load_rawdata(rawdata=rawdata)

    comp = BDICompressor(stream=stream, bandwidth=8, wordbitwidth=8)
    cratio = comp.calc_compression_ratio(verbose=2)
    print(f"total compression ratio: {cratio:.6f}")