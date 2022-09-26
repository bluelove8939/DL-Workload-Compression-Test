from typing import Callable
import numpy as np

from compression.binary_array import array2binary
from compression.algorithms import bdizv_compression, bdizv_decompression


class MemoryModule(object):
    def __init__(self, chunksize: int=128, wordwidth: int=32, dtype: str='float32', cmethod: Callable or None=None):
        self.chunksize = chunksize
        self.wordwidth = wordwidth
        self.dtype = np.dtype(dtype)
        self.cmethod = cmethod

        self.body = ''

    def reset(self):
        self.body = ''

    def load(self, addr):
        pass