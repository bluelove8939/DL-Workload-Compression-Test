import os
import numpy as np
import matplotlib.pyplot as plt

from utils.analyzer import CSVAnalyzer


if __name__ == '__main__':
    ax = plt.axes()

    result_files = [
        os.path.join(os.curdir, 'logs', 'accelerator_cycle.csv'),
        os.path.join(os.curdir, 'logs', 'accelerator_cycle_quant.csv'),
    ]

    testbenches = {
        # Normal testbenches
        # VGG16
        f"{result_files[0]}_VGG16_features.2": 'VC1',
        f"{result_files[0]}_VGG16_features.7": 'VC2',
        f"{result_files[0]}_VGG16_features.12": 'VC3',
        f"{result_files[0]}_VGG16_features.19": 'VC4',

        # ResNet50
        f"{result_files[0]}_ResNet50_layer1.0.conv2": 'RC1',
        f"{result_files[0]}_ResNet50_layer2.3.conv2": 'RC2',
        f"{result_files[0]}_ResNet50_layer3.5.conv2": 'RC3',
        f"{result_files[0]}_ResNet50_layer4.2.conv2": 'RC4',

        # AlexNet
        f"{result_files[0]}_AlexNet_features.3": 'AC1',
        f"{result_files[0]}_AlexNet_features.6": 'AC2',
        # f"{result_files[0]}_AlexNet_features.8":  'AC3',
        f"{result_files[0]}_AlexNet_features.10": 'AC3',

        # Quantized testbenches
        # ResNet50
        f"{result_files[1]}_ResNet50_layer1.0.conv2": 'QRC1',
        f"{result_files[1]}_ResNet50_layer2.3.conv2": 'QRC2',
        f"{result_files[1]}_ResNet50_layer3.5.conv2": 'QRC3',
        f"{result_files[1]}_ResNet50_layer4.2.conv2": 'QRC4',

        # GoogLeNet
        f"{result_files[1]}_GoogLeNet_inception3a.branch2.1.conv": 'QGC1',
        f"{result_files[1]}_GoogLeNet_inception3b.branch3.1.conv": 'QGC2',
        f"{result_files[1]}_GoogLeNet_inception4a.branch2.1.conv": 'QGC3',
        f"{result_files[1]}_GoogLeNet_inception4c.branch2.0.conv": 'QGC4',
    }

    headers = ['performance gain']

    category_filter = lambda cat: cat in testbenches.keys()

    analyzer = CSVAnalyzer(filepaths=result_files, header=True, category_col=2, dtype='float', sep='_',
                           colors=('#A2A2A2', 'darkolivegreen', 'teal', 'gray'))
    analyzer.parse_file(category_filter=category_filter)
    analyzer.category_conversion(mappings=testbenches)
    analyzer.results['performance gain'] = np.array(analyzer.results['dense cycles']) / np.array(analyzer.results['sparse cycles'])
    analyzer.analyze_csv_with_graph(ax=ax, xtic_rotation=45, annotate=False, headers=headers)

    # ax.set_ylim([0, 1])
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', color='gray')
    ax.tick_params(axis='y', which='both', color='white')

    # ax.set_xlabel('testbenches', fontsize=13)
    ax.set_ylabel('performance gain', fontsize=11, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ax.legend(loc='upper center', ncol=4, frameon=False, shadow=False, bbox_to_anchor=(0,0.85,1,0.2))

    ax.figure.set_size_inches(7, 2)

    plt.tight_layout()
    plt.savefig("G:\내 드라이브\ICEIC 2023\Fig7_accelerator_performance_gain.pdf",
                dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.show()