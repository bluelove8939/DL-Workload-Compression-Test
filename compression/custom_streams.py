import os
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

        with open(self._filepath, 'rb') as file:
            file.seek(self.cursor)
            content = file.read(size)
            arr = np.frombuffer(content, dtype=self.dtype)

        self.cursor += size
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
        element_num = int(size / self._rawdata.dtype.itemsize)
        if self.cursor + element_num > self.fullsize():
            return None

        arr = self._rawdata[self.cursor:self.cursor+element_num]
        self.cursor += element_num
        return arr

    def fullsize(self):
        return self._rawdata.shape[0]


class MemoryStream(CustomStream):
    SEEKEND: str = 'SEEKEND'  # end position (address -1)
    SEEKCUR: str = 'SEEKCUR'  # current position (current cursor)
    SEEKSTR: str = 'SEEKSTR'  # starting position (address 0)

    def __init__(self, cacheline_size: int, dtype: np.dtype) -> None:
        super(MemoryStream, self).__init__()
        self._storage: list = []
        self._cacheline_size: int = cacheline_size
        self.cursor: int = 0
        self.dtype = dtype
        self.name: str = "MemoryStream"

    def reset(self) -> None:
        self._storage = []
        self._memsize = 0
        self.cursor = 0

    def fetch(self, size: int) -> np.ndarray or None:
        if self.cursor + size > self.fullsize():
            return None

        content = bytes()
        for addr in range(self.cursor, self.cursor + size, 1):
            content += self._storage[addr]

        arr = np.frombuffer(content, dtype=self.dtype)
        self.cursor += size

        return arr

    def store(self, arr: np.ndarray, store_type: str=SEEKEND) -> int or None:
        addr = self.cursor
        if store_type == MemoryStream.SEEKEND:
            addr = len(self._storage)
        elif store_type == MemoryStream.SEEKSTR:
            addr = 0

        while addr >= len(self._storage):
            self._storage.append(None)

        barr = arr.tobytes()
        if len(barr) > self._cacheline_size: return None
        self._storage[addr] = barr

        return addr

    def fullsize(self) -> int:
        return len(self._storage) * self._cacheline_size

    def move_address(self, addr: int) -> None:
        self.cursor = addr

    def curr_address(self) -> int:
        return self.cursor


if __name__ == '__main__':
    stream = MemoryStream(cacheline_size=8, dtype=np.dtype('int8'))
    for i in range(3):
        addr = stream.store(np.array([i] * 8, dtype=np.dtype('int8')), store_type=MemoryStream.SEEKEND)
        print(addr)
    for _ in range(3):
        arr = stream.fetch(size=1)
        print(arr)
    stream.move_address(addr=0)
    arr = stream.fetch(size=3)
    print(arr)