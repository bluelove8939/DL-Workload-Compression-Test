from typing import Callable

from compression.binary_array import print_binary, array2binary
from compression.algorithms import bitplane_compression, zrle_compression, zeroval_compression, ebp_compression
from compression.custom_streams import CustomStream, MemoryStream
from models.tools.progressbar import progressbar


# Compressor module
#
# Description
#   Module for simulating compression of data stream (from file or rawdata)
#   Data is fetched from the specific module 'CustomStream'

class Compressor(object):
    STOPCODE = 'STOPCODE'

    def __init__(self, cmethod: Callable, instream: CustomStream or None=None, outstream: MemoryStream or None=None, bandwidth: int=128, wordbitwidth: int=32):
        self.cmethod      = cmethod       # compression method (needs to follow the given interface)
        self.bandwidth    = bandwidth     # bandwidth in Bytes
        self.wordbitwidth = wordbitwidth  # wordwidth in bits
        self.instream     = instream      # to fetch chunk from data or file
        self.outstream    = outstream     # to load chunk to memory

    def register_input_stream(self, instream, bandwidth: int=None, wordbitwidth: int=None):
        self.instream     = instream
        self.bandwidth    = bandwidth    if bandwidth    is not None else self.bandwidth
        self.wordbitwidth = wordbitwidth if wordbitwidth is not None else self.wordbitwidth

    def step(self, verbose: int=1) -> str:
        stepsiz = int(self.bandwidth)
        arr = self.instream.fetch(size=stepsiz)

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
        cursor_limit = min(self.instream.fullsize(), self.bandwidth * maxiter) if maxiter != -1 else self.instream.fullsize()
        cntiter = 0

        self.instream.reset()

        while True:
            binarr = self.step(verbose)
            if binarr == Compressor.STOPCODE:
                return total_original_size / total_compressed_size if total_compressed_size != 0 else 0

            original_size = self.bandwidth * 8
            compressed_size = len(binarr)

            total_original_size += original_size
            total_compressed_size += compressed_size

            if verbose == 1:
                print(f"\r{progressbar(status=self.instream.cursor, total=cursor_limit, scale=50)}  "
                      f"cursor: {self.instream.cursor}/{self.instream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {original_size / compressed_size:.6f} "
                      f"({total_original_size / total_compressed_size:.6f})", end='          ')
            elif verbose == 2:
                print(f"\rcursor: {self.instream.cursor}/{self.instream.fullsize()} "
                      f"(maxiter {'N/A' if maxiter is None else maxiter})  "
                      f"compression ratio: {self.bandwidth * 8 / len(binarr):.6f} "
                      f"({total_original_size / total_compressed_size:.6f})")

            cntiter += 1
            if maxiter != -1 and cntiter >= maxiter:
                break

        return total_original_size / total_compressed_size

class BitPlaneCompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(BitPlaneCompressor, self).__init__(cmethod=bitplane_compression,
                                                 instream=instream,
                                                 outstream=outstream,
                                                 bandwidth=bandwidth,
                                                 wordbitwidth=wordbitwidth)

class ZeroRLECompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(ZeroRLECompressor, self).__init__(cmethod=zrle_compression,
                                                instream=instream,
                                                outstream=outstream,
                                                bandwidth=bandwidth,
                                                wordbitwidth=wordbitwidth)

class ZeroValueCompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(ZeroValueCompressor, self).__init__(cmethod=zeroval_compression,
                                                  instream=instream,
                                                  outstream=outstream,
                                                  bandwidth=bandwidth,
                                                  wordbitwidth=wordbitwidth)

class EBPCompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(EBPCompressor, self).__init__(cmethod=ebp_compression,
                                            instream=instream,
                                            outstream=outstream,
                                            bandwidth=bandwidth,
                                            wordbitwidth=wordbitwidth)