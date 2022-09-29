from typing import Callable
import numpy as np
import matplotlib.axes as axes
import matplotlib.pyplot as plt


class CSVAnalyzer(object):
    def __init__(self, filepath: str, header: bool=True, category_col: int=1, dtype: str='float', sep: str='_'):
        self.filepath = filepath
        self.header = header
        self.category_col = category_col
        self.dtype = dtype
        self.sep = sep

        self.categories: list or None = None
        self.results: dict or None= None

    def line_splitter(self, line: list) -> (str, list):
        return self.sep.join(line[:self.category_col]),  np.array(line[self.category_col:], dtype=np.dtype(self.dtype))

    def parse_file(self, category_filter: Callable=lambda x: True) -> None:
        self.categories = []
        self.results = {}

        with open(self.filepath, 'rt') as file:
            content = list(map(lambda x: x.split(','), file.readlines()))

            if self.header:
                header = list(map(lambda x: x.strip(), content[0][self.category_col:]))
                content = content[1:]
            else:
                header = list(range(content[0] - self.category_col))

            for head in header:
                self.results[head] = []

            for line in content:
                cat, dat = self.line_splitter(line)

                if not category_filter(cat):
                    continue

                self.categories.append(cat)
                for didx, d in enumerate(dat):
                    self.results[header[didx]].append(d)

    def analyze_csv_with_graph(self, ax: axes.Axes, xtic_rotation: int=0, annotate: bool=False) -> axes.Axes:
        if self.categories is None or self.results is None:
            self.parse_file()

        width_max = 0.8
        width = width_max / len(self.results.keys())

        x_axis = np.arange(len(self.categories))
        for idx, (key, val) in enumerate(self.results.items()):
            xval = x_axis + ((idx - (len(self.results.keys()) / 2) + 0.5) * width)
            ax.bar(xval, val, width=width, label=key)

            if annotate:
                for i, j in zip(xval, val):
                    ax.annotate(f"{j:.2f}", xy=(i, j + 0.05), ha='center')
        ax.set_xticks(x_axis, self.categories, rotation=xtic_rotation, ha='center')

        return ax