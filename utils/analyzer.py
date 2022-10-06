from typing import Callable
import numpy as np
import matplotlib.axes as axes
import matplotlib.pyplot as plt


class CSVAnalyzer(object):
    def __init__(self, filepaths: str or list, header: bool=True, category_col: int=1, dtype: str='float', sep: str='_',
                 colors: tuple=('gray', 'darkorange', 'olivedrab', 'steelblue', 'blue')):

        self.filepaths: list   = filepaths if isinstance(filepaths, list) else [filepaths]
        self.header: bool      = header
        self.category_col: int = category_col
        self.dtype: str        = dtype
        self.sep: str          = sep
        self.colors: tuple     = colors

        self.categories: list or None = None
        self.results: dict or None    = None

    def line_splitter(self, line: list) -> (str, list):
        return self.sep.join(line[:self.category_col]),  np.array(line[self.category_col:], dtype=np.dtype(self.dtype))

    def category_conversion(self, mappings: dict) -> None:
        for idx, cat in enumerate(self.categories):
            if cat in mappings.keys():
                self.categories[idx] = mappings[cat]

    def parse_file(self, category_filter: Callable=lambda x: True, header_filter: Callable=lambda x: True) -> None:
        self.categories = []
        self.results = {}

        for filepath in self.filepaths:
            with open(filepath, 'rt') as file:
                content = list(map(lambda x: x.split(','), file.readlines()))

                if self.header:
                    header = list(map(lambda x: x.strip(), content[0][self.category_col:]))
                    content = content[1:]
                else:
                    header = list(range(content[0] - self.category_col))

                for head in header:
                    if head not in self.results.keys() and header_filter(head):
                        self.results[head] = []

                for line in content:
                    cat, dat = self.line_splitter(line)
                    cat = filepath + '_' + cat

                    if not category_filter(cat):
                        continue

                    self.categories.append(cat)
                    for didx, d in enumerate(dat):
                        if header_filter(header[didx]):
                            self.results[header[didx]].append(d)

    def analyze_csv_with_graph(self, ax: axes.Axes, xtic_rotation: int=0, annotate: bool=False,
                               headers: list or None=None) -> axes.Axes:

        if self.categories is None or self.results is None:
            self.parse_file()

        if headers is None:
            headers = self.results.keys()

        width_max = 0.8
        width = width_max / len(headers)

        x_axis = np.arange(len(self.categories))

        for idx, key in enumerate(headers):
            val = self.results[key]
            xval = x_axis + ((idx - (len(headers) / 2) + 0.5) * width)
            ax.bar(xval, val, width=width, label=key, color=self.colors[idx])

            if annotate:
                for i, j in zip(xval, val):
                    ax.annotate(f"{j:.2f}", xy=(i, j + 0.05), ha='center')
        ax.set_xticks(x_axis, self.categories, rotation=xtic_rotation, ha='center' if xtic_rotation == 0 else 'right')

        return ax