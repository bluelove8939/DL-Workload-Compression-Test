from typing import Callable

from compression.binary_array import print_binary, array2binary
from compression.algorithms import bitplane_compression, zrle_compression, zeroval_compression, ebp_compression, zlib_compression, bdi_compression, ebdi_compression
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

    def calc_compression_ratio(self, maxiter: int=-1, verbose: int=1, verbose_step: int=1) -> float:
        original_size = (self.bandwidth * 8) if self.bandwidth != -1 else (self.instream.fullsize() * 8)
        compressed_size = 0
        total_original_size = 0
        total_compressed_size = 0
        cntiter = 0

        self.instream.reset()
        self._print_compr_verbose(verbose, scale=50, maxiter=maxiter,
                                  orig_size=0, comp_size=0,
                                  total_orig_size=0, total_comp_size=0)

        while True:
            binarr = self.step(verbose)
            if binarr == Compressor.STOPCODE:
                break

            compressed_size = len(binarr)

            total_original_size += original_size
            total_compressed_size += compressed_size

            if cntiter % verbose_step == 0:
                self._print_compr_verbose(verbose, scale=50, maxiter=maxiter,
                                          orig_size=original_size, comp_size=compressed_size,
                                          total_orig_size=total_original_size, total_comp_size=total_compressed_size)

            cntiter += 1
            if maxiter != -1 and cntiter >= maxiter:
                break

        self._print_compr_verbose(verbose, scale=50, maxiter=maxiter,
                                  orig_size=original_size, comp_size=compressed_size,
                                  total_orig_size=total_original_size, total_comp_size=total_compressed_size)

        return total_original_size / (total_compressed_size + 1e-10)

    def _print_compr_verbose(self, verbose: int, scale: int=50, maxiter: int=-1, orig_size: float=0, comp_size: float=0,
                             total_orig_size: float=0, total_comp_size: float=0) -> None:

        cursor_limit = self.instream.fullsize()
        if maxiter != -1 and self.bandwidth != -1:
            cursor_limit = maxiter * self.bandwidth

        if verbose == 1:
            print(f"\r{progressbar(status=self.instream.cursor, total=cursor_limit, scale=scale)}  "
                  f"cursor: {self.instream.cursor}/{self.instream.fullsize()} "
                  f"({self.instream.cursor / self.instream.fullsize() * 100:6.2f}%)  "
                  f"compression ratio: {orig_size / (comp_size + 1e-6):.6f} "
                  f"({total_orig_size / (total_comp_size + 1e-6):.6f})", end='          ')
        elif verbose == 2:
            print(f"\rcursor: {self.instream.cursor}/{self.instream.fullsize()} "
                  f"({self.instream.cursor / self.instream.fullsize() * 100:6.2f}%)  "
                  f"compression ratio: {orig_size / (comp_size + 1e-6):.6f} "
                  f"({total_orig_size / (total_comp_size + 1e-6):.6f})")


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

class ZlibCompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(ZlibCompressor, self).__init__(cmethod=zlib_compression,
                                             instream=instream,
                                             outstream=outstream,
                                             bandwidth=bandwidth,
                                             wordbitwidth=wordbitwidth)

class BDICompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(BDICompressor, self).__init__(cmethod=bdi_compression,
                                            instream=instream,
                                            outstream=outstream,
                                            bandwidth=bandwidth,
                                            wordbitwidth=wordbitwidth)

class EBDICompressor(Compressor):
    def __init__(self, instream: CustomStream or None=None, outstream: MemoryStream or None=None,
                 bandwidth: int=128, wordbitwidth: int=32) -> None:
        super(EBDICompressor, self).__init__(cmethod=ebdi_compression,
                                             instream=instream,
                                             outstream=outstream,
                                             bandwidth=bandwidth,
                                             wordbitwidth=wordbitwidth)