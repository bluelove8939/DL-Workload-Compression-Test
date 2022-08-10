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
        num = (2 ** wordwidth) - num
    num = '0' + bin(num)[2:]
    return num.rjust(wordwidth, num[0])

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

def binary2array(binarr: str, wordwidth: int) -> np.ndarray:
    arr = np.array([binary2integer(binarr[i:i+wordwidth], wordwidth) for i in range(0, len(binarr), wordwidth)])
    return arr

def print_binary(binstr: str, swidth: int=8, startswith='', endswith='\n') -> None:
    print(startswith, end='')
    for i in range(0, len(binstr), swidth):
        print(binstr[i:i+swidth], end=' ')
    print(endswith, end='')