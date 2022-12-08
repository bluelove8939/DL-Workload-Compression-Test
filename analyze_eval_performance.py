import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from simulation.testbenches import testbenches
from utils.analyzer import CSVAnalyzer


parser = argparse.ArgumentParser(description='Analysis Configs')
parser.add_argument('-f', '--filename', default='eval_activation_compression_quant_pruned',
                    help='Name of the result file (without extension)', dest='filename')
args, _ = parser.parse_known_args()


if __name__ == '__main__':
    # Path
    filename = args.filename
    result_file = os.path.join(os.curdir, 'logs', f'{filename}.csv')
    image_dirname = os.path.join('G:', '내 드라이브', 'ESL2023')
    image_filename = f'Fig_{filename}.pdf'

    os.makedirs(image_dirname, exist_ok=True)

    # Draw the figure
    ax = plt.axes()

    headers = ['ratio']
    tb_mappings = {'_'.join(key):val for key, val in testbenches.items()}
    category_filter = lambda cat: cat in tb_mappings.keys()

    analyzer = CSVAnalyzer(filepaths=result_file, header=True, category_col=2, dtype='float', sep='_',
                           colors=('#A2A2A2', '#EEEEEE'),
                           hatches=(None, None))
    analyzer.parse_file(category_filter=category_filter)
    analyzer.category_conversion(mappings=tb_mappings)

    width_max = 0.8
    width = width_max / len(headers)

    x_axis = np.arange(len(analyzer.categories))

    val = np.array(analyzer.results['ratio'])
    xval = x_axis + ((0 - (len(headers) / 2) + 0.5) * width)
    ax.bar(xval, val, width=width, label='effectual', color=analyzer.colors[0], hatch=analyzer.hatches[0],
           edgecolor='black', linewidth=0.5)
    for i, j in zip(xval, val):
        ax.annotate(f"{j:.2f}", xy=(i, j + 0.05), ha='center', size=8)
    ax.bar(xval, 1 - val, width=width, label='ineffectual', color=analyzer.colors[1], hatch=analyzer.hatches[1],
           edgecolor='black', linewidth=0.5, bottom=val)
    ax.set_xticks(x_axis, analyzer.categories, rotation=45, ha='right')

    ax.set_ylim([0, 1])
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', color='gray')
    ax.tick_params(axis='y', which='both', color='white')

    # ax.set_xlabel('testbenches', fontsize=13)
    ax.set_ylabel('ratio', fontsize=11, fontweight='bold')

    ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.legend(loc='upper center', ncol=4, frameon=False, shadow=False, bbox_to_anchor=(0,1.2,1,0.2))

    ax.figure.set_size_inches(7, 2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dirname, image_filename), dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.show()