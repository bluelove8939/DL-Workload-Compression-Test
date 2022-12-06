import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, '../../logs', 'entropy_test.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()


if __name__ == '__main__':
    target_modelnames = ['AlexNet', 'VGG16']
    # targets = ['ReLU', 'Conv2d']
    targets = ['ReLU']
    maxdepths = [7, 15]

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})

    for target_modelname, maxdepth, axis in zip(target_modelnames, maxdepths, axes):
        categories = np.array(list(range(1, maxdepth + 1, 1)))  # x-axis by depth
        results = {}

        logfilepath = comp_args.filepath

        with open(logfilepath, 'rt') as file:
            content = list(map(lambda x: x.split(','), file.readlines()))
            for target_layer_type in targets:
                results[target_layer_type] = np.array([0] * maxdepth, dtype=np.dtype('float32'))

            for modelname, filename, entropy in content:
                modelname = modelname.split('_')[0]
                layer_type, depth, outputname = filename.split('_')

                if target_modelname != modelname or layer_type not in results.keys():
                    continue

                if int(depth) < maxdepth:
                    results[layer_type][int(depth)] += float(entropy)

            print(results)

            for target_layer_type in targets:
                results[target_layer_type] /= 3.0

        width_max = 0.8
        width = width_max / len(results.keys())

        x_axis = np.arange(len(categories))
        for idx, (key, val) in enumerate(results.items()):
            xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
            axis.bar(xval, val, width=width, label=key)
            for i, j in zip(xval, val):
                axis.annotate(f"{j:.2f}", xy=(i, j + 0.1), ha='center', size=7)
        axis.set_xticks(x_axis, categories, rotation=0, ha='center')
        # axis.set_ylim([0.0, 1.0])

        axis.set_xlabel('depth')
        axis.set_ylabel('entropy')

        axis.set_title(f"Entropy of layers by depth ({target_modelname})")
        axis.legend()

    plt.tight_layout()
    plt.show()