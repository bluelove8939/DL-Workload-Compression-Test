import abc

from systempy.module import Module

from simulation.accelerators.systolic_array import SystolicArray
from simulation.accelerators.compressed_accelerator import CompressedAccelerator


class AcceleratorConfig(metaclass=abc.ABCMeta):
    def __init__(self):
        super(AcceleratorConfig, self).__init__()

    @abc.abstractmethod
    def generate(self) -> Module:
        pass


class CompressedAcceleratorConfig(AcceleratorConfig):
    def __init__(self, engine_num: int, pe_num: int, mult_num: int, chunk_size: int, fifo_capacity: int):
        super(CompressedAcceleratorConfig, self).__init__()

        self.engine_num = engine_num
        self.mult_num = mult_num
        self.pe_num = pe_num
        self.chunk_size = chunk_size
        self.fifo_capacity = fifo_capacity

    def generate(self) -> CompressedAccelerator:
        return CompressedAccelerator(mult_num=self.mult_num, pe_num=self.pe_num, chunk_size=self.chunk_size,
        fifo_capacity=self.fifo_capacity).compile(verbose=False)


class SystolicArrayWSConfig(AcceleratorConfig):
    def __init__(self, sa_shape: tuple[int, int]):
        super(SystolicArrayWSConfig, self).__init__()

        self.sa_shape = sa_shape

    def generate(self) -> Module:
        return SystolicArray(width=self.sa_shape[1], height=self.sa_shape[0])