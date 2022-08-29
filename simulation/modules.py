import abc
from typing import Iterable

from compression.modules import Compressor as _Compressor


class Module(metaclass=abc.ABCMeta):
    def __init__(self):
        super(Module, self).__init__()

        self._port_in = []
        self._port_out = []
        self.cycle = -1

    def link(self, other):
        self._port_out = other
        other._port_in = self

    @abc.abstractmethod
    def trigger(self):
        pass


class CompressorModule(Module):
    def __init__(self, compressor: _Compressor):
        super(CompressorModule, self).__init__()

        self.compressor = compressor

    def trigger(self):
        pass