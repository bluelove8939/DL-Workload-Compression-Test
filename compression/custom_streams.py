import os
import math
import logging
import abc

import numpy as np


class CustomStream(metaclass=abc.ABCMeta):
    def __init__(self):
        self.cursor = None
        self.name = "CustomStream"

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def fetch(self, size: int):
        pass

    @abc.abstractmethod
    def fullsize(self):
        pass

    def __str__(self):
        return self.name


class FileStream(CustomStream):
    def __init__(self, filepath: str=None, dtype: np.dtype=None) -> None:
        super(FileStream, self).__init__()
        self._filepath = filepath
        self.dtype = dtype
        self.cursor = 0
        self.name = f"FileStream: {filepath}"

    def load_filepath(self, filepath: str, dtype: np.dtype) -> None:
        self._filepath = filepath
        self.dtype = dtype
        self.reset()
        self.name = f"FileStream: {filepath}"

    def reset(self) -> None:
        self.cursor = 0

    def fetch(self, size: int) -> np.ndarray or None:
        if self.cursor + size > self.fullsize():
            return None

        if size == -1 and self.cursor > 0:
            return None

        # size = self.fullsize() if size == -1 else size

        with open(self._filepath, 'rb') as file:
            file.seek(self.cursor)
            content = file.read(size)
            arr = np.frombuffer(content, dtype=self.dtype)

        self.cursor += size if size != -1 else self.fullsize()
        return arr

    def fullsize(self):
        return os.path.getsize(self._filepath)


class DataStream(CustomStream):
    def __init__(self, rawdata: np.ndarray=None) -> None:
        super(DataStream, self).__init__()
        self._rawdata = rawdata
        self.cursor = 0
        self.name = "DataStream"

    def load_rawdata(self, rawdata: np.ndarray) -> None:
        self._rawdata = rawdata
        self.reset()

        if len(self._rawdata.shape) > 1:
            logging.warning(f'[Warning] Ambiguous because given raw data is not flat (shape: {self._rawdata.shape})')
            self._rawdata.flatten()

    def reset(self) -> None:
        self.cursor = 0

    def fetch(self, size: int) -> np.ndarray or None:
        if size == -1:
            return self._rawdata

        element_num = int(size / self._rawdata.dtype.itemsize)
        if self.cursor + element_num > self.fullsize():
            return None

        arr = self._rawdata[self.cursor:self.cursor+element_num]
        self.cursor += element_num
        return arr

    def fullsize(self):
        return self._rawdata.shape[0]


class MemoryStream(CustomStream):
    SEEKEND = 'SEEKEND'  # end position (address -1)
    SEEKCUR = 'SEEKCUR'  # current position (current cursor)
    SEEKSTR = 'SEEKSTR'  # starting position (address 0)

    def __init__(self) -> None:
        super(MemoryStream, self).__init__()
        self._storage: list = []
        self.cursor: int = 0
        self.name: str = "MemoryStream"

    def reset(self) -> None:
        self._storage = []
        self.cursor = 0

    def fetch(self, size: int) -> str or None:
        return ''.join(self._storage[self.cursor:self.cursor + size])

    def store(self, arr: str, store_type: str=SEEKEND) -> int or None:
        barr_size = math.ceil(len(arr) / 8)  # size of byte array



    def fullsize(self) -> int:
        return len(self._storage)

    def move_address(self, addr: int) -> None:
        self.cursor = addr

    def curr_address(self) -> int:
        return self.cursor


if __name__ == '__main__':
    pass