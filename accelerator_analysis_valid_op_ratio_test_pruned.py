import os
import numpy as np
import matplotlib.pyplot as plt

from utils.analyzer import CSVAnalyzer


if __name__ == '__main__':
    ax = plt.axes()

    result_files = {
        0 : os.path.join(os.curdir, 'logs', 'accelerator_performance.csv'),
        10: os.path.join(os.curdir, 'logs', 'accelerator_performance_pruned_10.csv'),
        30: os.path.join(os.curdir, 'logs', 'accelerator_performance_pruned_30.csv'),
        50: os.path.join(os.curdir, 'logs', 'accelerator_performance_pruned_50.csv'),
    }

    testbenches = {
        # Without pruning
        # ResNet50
        f"{result_files[0]}_ResNet50_layer1.0.conv2": 'RC1',
        f"{result_files[0]}_ResNet50_layer2.3.conv2": 'RC2',
        f"{result_files[0]}_ResNet50_layer3.5.conv2": 'RC3',
        f"{result_files[0]}_ResNet50_layer4.2.conv2": 'RC4',

        # AlexNet
        f"{result_files[0]}_AlexNet_features.3": 'AC1',
        f"{result_files[0]}_AlexNet_features.6": 'AC2',
        f"{result_files[0]}_AlexNet_features.10": 'AC3',

        # Pruning amount 10%
        # ResNet50
        f"{result_files[10]}_ResNet50_layer1.0.conv2": 'RC1',
        f"{result_files[10]}_ResNet50_layer2.3.conv2": 'RC2',
        f"{result_files[10]}_ResNet50_layer3.5.conv2": 'RC3',
        f"{result_files[10]}_ResNet50_layer4.2.conv2": 'RC4',

        # AlexNet
        f"{result_files[10]}_AlexNet_features.3":  'AC1',
        f"{result_files[10]}_AlexNet_features.6":  'AC2',
        f"{result_files[10]}_AlexNet_features.10": 'AC3',

        # Pruning amount 30%
        # ResNet50
        f"{result_files[30]}_ResNet50_layer1.0.conv2": 'RC1',
        f"{result_files[30]}_ResNet50_layer2.3.conv2": 'RC2',
        f"{result_files[30]}_ResNet50_layer3.5.conv2": 'RC3',
        f"{result_files[30]}_ResNet50_layer4.2.conv2": 'RC4',

        # AlexNet
        f"{result_files[30]}_AlexNet_features.3": 'AC1',
        f"{result_files[30]}_AlexNet_features.6": 'AC2',
        f"{result_files[30]}_AlexNet_features.10": 'AC3',

        # Pruning amount 50%
        # ResNet50
        f"{result_files[50]}_ResNet50_layer1.0.conv2": 'RC1',
        f"{result_files[50]}_ResNet50_layer2.3.conv2": 'RC2',
        f"{result_files[50]}_ResNet50_layer3.5.conv2": 'RC3',
        f"{result_files[50]}_ResNet50_layer4.2.conv2": 'RC4',

        # AlexNet
        f"{result_files[50]}_AlexNet_features.3": 'AC1',
        f"{result_files[50]}_AlexNet_features.6": 'AC2',
        f"{result_files[50]}_AlexNet_features.10": 'AC3',
    }

    results = {}
    categories = []
    headers = []

    category_filter = lambda cat: cat in testbenches.keys()

    for pamount, filename in result_files.items():
        analyzer = CSVAnalyzer(filepaths=filename, header=True, category_col=2, dtype='float', sep='_',
                               colors=('#A2A2A2', '#EEEEEE'),
                               hatches=(None, None))
        analyzer.parse_file(category_filter=category_filter, cat_file=True)
        analyzer.category_conversion(mappings=testbenches)
        categories = analyzer.categories

        tmp = 1 - np.array(analyzer.results['valid']) / np.array(analyzer.results['total'])

        results[pamount] = tmp
        headers.append(pamount)

    colors = ('black', '#A2A2A2', 'white', 'white')
    hatches = (None, None, None, '//////')

    # print(results)

    width_max = 0.6
    width = width_max / len(headers)

    x_axis = np.arange(len(categories))

    for idx, key in enumerate(headers):
        val = results[key]
        xval = x_axis + ((idx - (len(headers) / 2) + 0.5) * width)
        ax.bar(xval, val, width=width, label=f"{key}%", color=colors[idx], hatch=hatches[idx],
               edgecolor='black', linewidth=0.5)

        # if annotate:
        #     for i, j in zip(xval, val):
        #         ax.annotate(f"{j:.2f}", xy=(i, j + 0.05), ha='center')

    ax.set_xticks(x_axis, categories, rotation=45, ha='right')

    ax.set_ylim([0, 1])
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', color='gray')
    ax.tick_params(axis='y', which='both', color='white')

    # ax.set_xlabel('testbenches', fontsize=13)
    ax.set_ylabel('performance gain', fontsize=11, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(loc='upper center', ncol=4, frameon=False, shadow=False, bbox_to_anchor=(0, 0.9, 1, 0.2))

    ax.figure.set_size_inches(7, 2.5)

    plt.tight_layout()
    # plt.savefig("G:\내 드라이브\ICEIC 2023\Fig_accelerator_performance_gain_pruned.pdf",
    #             dpi=200, bbox_inches='tight', pad_inches=0)
    plt.show()