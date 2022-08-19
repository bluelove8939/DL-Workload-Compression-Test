import math

import numpy as np


# Functions for managing value array and its binary representation
#
# References
#   - Bit-Plane Compression: Transforming Data for Better Compression in Many-core Architectures, ISCA2016
#   - https://github.com/lukasc-ch/ExtendedBitPlaneCompression/tree/master/algoEvals

def binary_shrinkable(num: str, d: int) -> bool:
    for i in range(len(num)-d):
        if num[0] != num[i]:
            return False
    return True

def binary_xor(a: str, b: str) -> str:
    return bin(int(a, 2) ^ int(b, 2))[2:].zfill(len(a))

def binary_not(a: str) -> str:
    return ''.join(['1' if letter == '0' else '0' for letter in a])

def binary_and(a: str, b: str) -> str:
    return bin(int(a, 2) & int(b, 2))[2:].zfill(len(a))

def binary_or(a: str, b: str) -> str:
    return bin(int(a, 2) | int(b, 2))[2:].zfill(len(a))

def integer2binary(num: int, wordwidth: int):  # 2's complement number conversion
    if num < 0:
        num = (2 ** wordwidth) - abs(num)
    binnum = bin(num)[2:]
    return binnum.rjust(wordwidth, '0' if num >= 0 else '1')

def binary2integer(num: str, wordwidth: int):  # 2's complement number conversion
    sign = num[0]
    num = int(num, 2)

    if sign == '1':
        return num - (2 ** wordwidth)
    return num

def array2binary(arr: np.ndarray, wordwidth: int=None) -> str:
    barr = arr.byteswap().tobytes()
    precision = arr.dtype.itemsize * 8
    if wordwidth is None:
        wordwidth = precision
    rawbinarr = bin(int.from_bytes(barr, byteorder='big'))[2:].zfill(len(barr) * 8)
    binarr = ''.join([rawbinarr[i:i + precision][-wordwidth:] if wordwidth <= precision
                      else rawbinarr[i:i + precision].rjust(wordwidth, rawbinarr[i:i + precision][0])
                      for i in range(0, len(rawbinarr), precision)])
    return binarr

def binary2array(binarr: str, wordwidth: int, dtype: np.dtype) -> np.ndarray:
    bytearr = bytearray()
    for i in range(0, len(binarr), wordwidth):
        bytearr += int(binarr[i:i+wordwidth], 2).to_bytes(dtype.itemsize, byteorder='big')
    arr = np.frombuffer(bytearr, dtype=dtype).byteswap()
    return arr

def array_caster(arr: np.ndarray, dtype: np.dtype):
    return np.frombuffer(arr.tobytes(), dtype=dtype)

def binary_caster(binnum: str, dtype: np.dtype):
    if 'float' in dtype.name:
        return np.frombuffer(int(binnum, 2).to_bytes(len(binnum) // 8, byteorder='big'), dtype=dtype)
    return np.array([binary2integer(binnum, len(binnum))], dtype=dtype)

def print_binary(binstr: str, swidth: int=8, startswith='', endswith='\n') -> None:
    print(startswith, end='')
    for i in range(0, len(binstr), swidth):
        print(binstr[i:i+swidth], end=' ')
    print(endswith, end='')


# Methods for BDI

def trunc_array2binary(arr: np.ndarray, wordwidth: int=None) -> tuple:
    binarr = array2binary(arr, wordwidth)
    binnum_arr = [binarr[i:i+wordwidth] for i in range(0, len(binarr), wordwidth)]
    mincnt = wordwidth

    for binnum in binnum_arr:
        cnt = 0

        for let in binnum[1:]:
            if let == binnum[0]: cnt += 1
            else: break

        mincnt = min(mincnt, cnt)

    return wordwidth - mincnt, ''.join(list(map(lambda x: x[mincnt:], binnum_arr)))


if __name__ == '__main__':
    num = 14
    print(bin(num))
    print(integer2binary(num, 8))
    print(binary2integer(integer2binary(num, 8), 8))

    print(array2binary(np.array(-2), 8))