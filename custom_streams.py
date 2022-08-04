import os
import logging
import abc

import numpy as np


class CustomStream(metaclass=abc.ABCMeta):
    def __init__(self):
        self.cursor = None

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def fetch(self, size: int):
        pass

    @abc.abstractmethod
    def fullsize(self):
        pass


class FileStream(CustomStream):
    def __init__(self, filepath: str=None, dtype: np.dtype=None) -> None:
        super(FileStream, self).__init__()
        self.filepath = filepath
        self.dtype = dtype
        self.cursor = 0

    def load_filepath(self, filepath: str, dtype: np.dtype) -> None:
        self.filepath = filepath
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.cursor = 0

    def fetch(self, size: int) -> np.ndarray or None:
        if self.cursor + size > self.fullsize():
            return None

        with open(self.filepath, 'rb') as file:
            file.seek(self.cursor)
            content = file.read(size)
            arr = np.frombuffer(content, dtype=self.dtype)

        self.cursor += size
        return arr

    def fullsize(self):
        return os.path.getsize(self.filepath)


class DataStream(CustomStream):
    def __init__(self, rawdata: np.ndarray=None) -> None:
        super(DataStream, self).__init__()
        self._rawdata = rawdata
        self.cursor = 0

    def load_rawdata(self, rawdata: np.ndarray) -> None:
        self._rawdata = rawdata
        self.reset()

        if len(self._rawdata.shape) > 1:
            logging.warning(f'[Warning] Ambiguous because given raw data is not flat (shape: {self._rawdata.shape})')
            self._rawdata.flatten()

    def reset(self) -> None:
        self.cursor = 0

    def fetch(self, size: int) -> np.ndarray or None:
        if self.cursor + size > self.fullsize():
            return None

        arr = self._rawdata[self.cursor:self.cursor+size]
        self.cursor += size
        return arr

    def fullsize(self):
        return self._rawdata.shape[0]