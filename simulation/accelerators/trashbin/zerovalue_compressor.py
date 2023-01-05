import numpy as np

from systempy.module import Module
from systempy.elements import InputPort, OutputPort, ClockPort, Register, Wire


class PrefixSumUnit(Module):
    def __init__(self, length=10):
        super(PrefixSumUnit, self).__init__()

        self.length = length

        self.mask = InputPort(shape=self.length, dtype='int32')
        self.psum = OutputPort(shape=self.length, dtype='int32')

        self.psum_reg = Register(shape=self.length, dtype='int32')
        self.psum.assign(self.psum_reg)

    @Module.always('mask')
    def main(self):
        mask = self.mask.get_raw()
        self.psum_reg << np.array([np.sum(mask[:i]) for i in range(len(mask))])


class BubbleCollapseShifter(Module):
    def __init__(self):
        super(BubbleCollapseShifter, self).__init__()

        self.psum = InputPort()
        self.line = InputPort()

        self.compr = OutputPort()
        self.compr_reg = Register()

        self.compr.assign(self.compr_reg)

    @Module.always('psum', 'line')
    def main(self):
        original = np.array(self.line.get_raw())
        compressed = np.zeros_like(original)
        for idx, num in enumerate(original[np.nonzero(original)]):
            compressed[idx] = num

        self.compr_reg << compressed


class ZVCompressor(Module):
    def __init__(self, length=10):
        super(ZVCompressor, self).__init__()

        self.length = length

        self.clk = ClockPort()
        self.reset_n = InputPort()

        self.orig_line = InputPort(shape=self.length, dtype='int32')
        self.comp_line = OutputPort(shape=self.length, dtype='int32')

        # Stage 1: prefix sum
        self.mask = Register(shape=self.length, dtype='int32')
        self.psum_wire = Wire(shape=self.length, dtype='int32')
        self.ps_unit = PrefixSumUnit(length=length).assign(mask=self.mask, psum=self.psum_wire)

        self.psum_pipe1 = Register(shape=self.length, dtype='int32')
        self.orig_pipe1 = Register(shape=self.length, dtype='int32')

        # Stage 2: shifting and compress
        self.comp_wire = Wire(shape=self.length, dtype='int32')
        self.comp_reg = Register(shape=self.length, dtype='int32')
        self.bc_shifter = BubbleCollapseShifter().assign(psum=self.psum_pipe1, line=self.orig_pipe1, compr=self.comp_wire)

        self.comp_line.assign(self.comp_reg)

    @Module.always('orig_line')
    def mask_generation(self):
        self.mask.set((self.orig_line.get_raw() != 0).astype(np.dtype('int32')))

    @Module.always('posedge clk', 'negedge reset_n')
    def main(self):
        if not self.reset_n.get():
            self.psum_pipe1 <<= np.zeros(self.length)
            self.orig_pipe1 <<= np.zeros(self.length)
            self.comp_reg   <<= np.zeros(self.length)
        else:
            self.psum_pipe1 <<= self.psum_wire.get()
            self.orig_pipe1 <<= self.orig_line.get()
            self.comp_reg   <<= self.comp_wire.get()


# Testbench for PrefixSumUnit
if __name__ == '__main__':
    psum_unit = PrefixSumUnit().compile(verbose=True)

    length = 10
    sparse = 5
    scale = 5

    for _ in range(3):
        data = (np.random.rand(length) * scale ).astype(dtype=np.dtype('int32'))
        sidx = (np.random.rand(sparse) * length).astype(dtype=np.dtype('int32'))
        data[sidx] = 0
        mask = (data != 0).astype(np.dtype('int32'))

        psum_unit.run(mask=mask)

        psum_unit.print_summary('psum')
        print("mask:", mask, '\n')

# Testbench for BubbleCollapseShifter
if __name__ == '__main__':
    bc_shifter = BubbleCollapseShifter().compile(verbose=True)

    length = 10
    sparse = 5
    scale = 5

    for _ in range(3):
        data = (np.random.rand(length) * scale).astype(dtype=np.dtype('int32'))
        sidx = (np.random.rand(sparse) * length).astype(dtype=np.dtype('int32'))
        data[sidx] = 0
        mask = (data != 0).astype(np.dtype('int32'))
        psum = np.array([np.sum(mask[:i+1]) for i in range(len(mask))])

        bc_shifter.run(psum=psum, line=data)

        bc_shifter.print_summary('compr')
        print("data: ", data, '\n')

# Testbench for ZVCompressor
if __name__ == '__main__':
    compression_unit = ZVCompressor().compile(verbose=True)

    length = 10
    sparse = 5
    scale = 5

    compression_unit.run(reset_n=0)
    compression_unit.run(reset_n=1)

    for _ in range(3):
        orig_line = (np.random.rand(length) * scale).astype(dtype=np.dtype('int32'))
        sidx = (np.random.rand(sparse) * length).astype(dtype=np.dtype('int32'))
        orig_line[sidx] = 0

        compression_unit.run(clk=0, orig_line=orig_line)
        compression_unit.run(clk=1)
        compression_unit.print_summary('orig_line', 'comp_line')

    for _ in range(3):
        compression_unit.run(clk=0)
        compression_unit.run(clk=1)
        compression_unit.print_summary('orig_line', 'comp_line')
