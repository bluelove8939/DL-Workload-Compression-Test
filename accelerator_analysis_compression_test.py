import os
import numpy as np
import matplotlib.pyplot as plt

from utils.analyzer import CSVAnalyzer


if __name__ == '__main__':
    ax = plt.axes()

    model_name = 'InceptionV3'
    quant = True

    file_name = 'accelerator_compression'
    fig_title = f'Accelerator Compression Test: {"quantized " if quant else ""}{model_name}'
    full_filename = f"{file_name}{'_quant' if quant else ''}.csv"

    category_filter = lambda x: model_name.lower() in x.lower()

    analyzer = CSVAnalyzer(filepath=os.path.join(os.curdir, 'logs', full_filename),
                           header=True, category_col=2, dtype='float', sep=' ')
    analyzer.parse_file(category_filter=category_filter)
    analyzer.analyze_csv_with_graph(ax=ax, xtic_rotation=90, annotate=False)

    ax.set_ylim([0, 4])
    ax.set_title(fig_title)
    ax.legend()

    plt.tight_layout()
    plt.show()