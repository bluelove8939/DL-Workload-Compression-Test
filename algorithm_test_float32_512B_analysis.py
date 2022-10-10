import os
import numpy as np
import matplotlib.pyplot as plt

from utils.analyzer import CSVAnalyzer


if __name__ == '__main__':
    ax = plt.axes()
    algorithms = ['BDI', 'ZVC', 'BDIZV', 'CSC']

    dirname = os.path.join(os.curdir, 'logs')
    filename = 'accelerator_algorithm_test_float32_512B.csv'

    colors = ('gray', 'darkorange', 'olivedrab', 'steelblue', 'blue')
    markers = ('o', 'x', '^')
    headers = ['BDIZV', 'ZRLE', 'CSC']

    analyzer = CSVAnalyzer(filepaths=os.path.join(dirname, filename), header=True, category_col=1, dtype='float', sep='_')
    analyzer.parse_file()
    # analyzer.category_conversion()
    # analyzer.analyze_csv_with_graph(ax=ax, xtic_rotation=45, annotate=False, headers=algorithms)

    # width_max = 0.8
    # width = width_max / len(headers)

    x_axis = np.arange(len(analyzer.categories))

    for idx, key in enumerate(headers):
        val = analyzer.results[key]
        xval = x_axis
        # print(len(xval), len(val))
        ax.plot(xval, val, label=key, color=colors[idx], marker=markers[idx])
    ax.set_xticks(x_axis, analyzer.categories, rotation=0, ha='center')

    # ax.set_ylim([0, 4.8])
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', color='gray')
    ax.tick_params(axis='y', which='both', color='white')

    # ax.set_xlabel('testbenches', fontsize=13)
    ax.set_ylabel('compression ratio', fontsize=11, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.legend(loc='upper center', ncol=4, frameon=False, shadow=False, bbox_to_anchor=(0,0.85,1,0.2))

    ax.figure.set_size_inches(7, 3)

    plt.tight_layout()
    plt.savefig("G:\내 드라이브\ICEIC 2023\Fig_algorithm_test_float32_512B.pdf",
                dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.show()