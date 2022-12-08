import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils.analyzer import CSVAnalyzer
from simulation.testbenches import testbenches


parser = argparse.ArgumentParser(description='Analysis Configs')
parser.add_argument('-f', '--filename', default='eval_activation_compression_pruned',
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

    algorithms = ['ZVC', 'BDIZV', 'ZRLE', 'CSC']
    tb_mappings = {'_'.join(key):val for key, val in testbenches.items()}
    category_filter = lambda cat: cat in tb_mappings.keys()

    analyzer = CSVAnalyzer(filepaths=result_file, header=True, category_col=2, dtype='float', sep='_',
                           colors=('black', '#A2A2A2', 'white', 'white'),
                           hatches=(None, None, None, '//////'))
    analyzer.parse_file(category_filter=category_filter)
    analyzer.category_conversion(mappings=tb_mappings)
    analyzer.analyze_csv_with_graph(ax=ax, xtic_rotation=45, annotate=False, headers=algorithms)

    # ax.set_ylim([0, 3.8])
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', color='gray')
    ax.tick_params(axis='y', which='both', color='white')

    # ax.set_xlabel('testbenches', fontsize=13)
    ax.set_ylabel('compression ratio', fontsize=11, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(loc='upper center', ncol=4, frameon=False, shadow=False, bbox_to_anchor=(0,1,1,0.2))

    ax.figure.set_size_inches(7, 2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dirname, image_filename), dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.show()