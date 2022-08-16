import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, 'logs', 'sparsity_test_quant_int8.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()


if __name__ == '__main__':
    # layer_types = ['sparsity', 'BatchNorm2D', 'Conv2D', 'ReLU']
    layer_types = ['ConvReLU2d']
    categories = []
    results = {}

    logfilepath = comp_args.filepath

    with open(logfilepath, 'rt') as file:
        content = list(map(lambda x: x.split(','), file.readlines()))

        for line in content:
            if line[0] not in categories:
                categories.append(line[0])

        for layer_type in layer_types:
            results[layer_type] = [0] * len(categories)
            results[f"{layer_type}_total"] = [0] * len(categories)

        for modelname, filename, arrsize, zerocnt in content:
            for layer_type in layer_types:
                if layer_type != 'sparsity' and layer_type.lower() not in filename.lower():
                    continue

                results[layer_type][categories.index(modelname)] += int(zerocnt)
                results[f'{layer_type}_total'][categories.index(modelname)] += int(arrsize)

        for layer_type in layer_types:
            results[layer_type] = np.array(results[layer_type]) / (np.array(results[f"{layer_type}_total"]) + 1e-4)
            results[layer_type][results[layer_type] == np.nan] = 0
            del results[f"{layer_type}_total"]

        categories = list(map(lambda x: x.split('_')[0], categories))

    width_max = 0.8
    width = width_max / len(results.keys())

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
        plt.bar(xval, val, width=width, label=key)
        for i, j in zip(xval, val):
            plt.annotate(f"{j:.2f}", xy=(i, j + 0.02), ha='center')
    plt.xticks(x_axis, categories, rotation=0, ha='center')
    plt.ylim([0.0, 1.0])

    plt.title("Sparsity of quantized CNN layers")
    plt.legend()
    plt.tight_layout()
    plt.show()