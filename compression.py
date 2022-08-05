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

def binary_xor(a: str, b: str) -> str:
    wordwidth = len(a)
    return bin(int(a, 2) ^ int(b, 2))[2:].zfill(wordwidth)

def array2binary(arr: np.ndarray, wordwidth: int=None) -> str:
    barr = arr.byteswap().tobytes()
    precision = arr.dtype.itemsize * 8
    if wordwidth is None:
        wordwidth = precision
    rawbinarr = bin(int.from_bytes(barr, byteorder='big'))[2:].zfill(len(barr) * 8)
    binarr = ''.join([rawbinarr[i:i + precision][-wordwidth:] if wordwidth <= precision
                      else rawbinarr[i:i + precision].zfill(wordwidth)
                      for i in range(0, len(rawbinarr), precision)])
    return binarr

def binary2array(binarr: str, wordwidth: int) -> np.ndarray:
    arr = np.array([binarr[i:i+wordwidth] for i in range(len(binarr))])
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
    diffs = [array2binary(d, wordwidth+1) for d in arr[1:] - arr[0:1]]
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
            encoded += '00010' + bin(dbx.find('11'))[2:].zfill(5)
        elif dbx.count('1') == 1:
            encoded += '00011' + bin(dbx.find('1'))[2:].zfill(5)
        else:
            encoded += '1' + dbx

    return encoded


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

    def calc_compression_ratio(self, maxiter: int=None, verbose: int=1) -> float:
        total_original_size = 0
        total_compressed_size = 0
        cntiter = 0

        self.stream.reset()

        if verbose:
            print(f"compression ratio test with {self.stream}")

        while True:
            binarr = self.step(verbose)
            if binarr == Compressor.STOPCODE:
                return total_original_size / total_compressed_size

            original_size = self.bandwidth * 8
            compressed_size = len(binarr)

            total_original_size += original_size
            total_compressed_size += compressed_size

            if verbose == 1:
                print(f"\rcursor: {self.stream.cursor}/{self.stream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {original_size / compressed_size:.6f} "
                      f"({total_original_size / total_compressed_size:.6f})", end='')
            elif verbose == 2:
                print(f"\rcursor: {self.stream.cursor}/{self.stream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {self.bandwidth * 8 / len(binarr):.6f} "
                      f"({total_original_size / total_compressed_size:.6f})")

            cntiter += 1
            if maxiter is not None and cntiter > maxiter:
                break

        return original_size / compressed_size


class BitPlaneCompressor(Compressor):
    def __init__(self, stream: CustomStream, bandwidth: int=128, wordbitwidth: int=32):
        super(BitPlaneCompressor, self).__init__(cmethod=bitplane_compression,
                                                 stream=stream,
                                                 bandwidth=bandwidth,
                                                 wordbitwidth=wordbitwidth)


if __name__ == '__main__':
    from custom_streams import DataStream

    stream = DataStream()
    stream.load_rawdata(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,], dtype=np.dtype(int)))

    comp = BitPlaneCompressor(stream=stream, bandwidth=128, wordbitwidth=32)
    cratio = comp.calc_compression_ratio(verbose=2)
    print(f"total compression ratio: {cratio:.6f}")