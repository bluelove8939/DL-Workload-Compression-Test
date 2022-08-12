import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, 'logs', 'entropy_test2.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()


if __name__ == '__main__':
    targets = [
        ('resnet18', 'batchnorm2d'),
        ('resnet18', 'relu'),
        ('resnet18', 'conv2d'),
    ]
    maxdepth = 20
    categories = np.array(list(range(1, maxdepth+1, 1)))  # x-axis by depth
    results = {}

    logfilepath = comp_args.filepath

    with open(logfilepath, 'rt') as file:
        content = list(map(lambda x: x.split(','), file.readlines()))
        for target_modelname, target_layer_type in targets:
            results[f"{target_modelname} {target_layer_type}"] = np.array([0] * maxdepth, dtype=np.dtype('float32'))

        for modelname, filename, entropy in content:
            modelname = modelname.split('_')[0]
            layer_type, depth, outputname = filename.split('_')

            if f"{modelname.lower()} {layer_type.lower()}" not in results.keys():
                continue

            if int(depth) < maxdepth:
                results[f"{modelname.lower()} {layer_type.lower()}"][int(depth)] += float(entropy)

        print(results)

        for target_modelname, target_layer_type in targets:
            results[f"{target_modelname} {target_layer_type}"] /= 3.0

    width_max = 0.8
    width = width_max / len(results.keys())

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
        plt.bar(xval, val, width=width, label=key)
        # for i, j in zip(xval, val):
        #     plt.annotate(f"{j:.2f}", xy=(i, j + 0.2), ha='center')
    plt.xticks(x_axis, categories, rotation=0, ha='center')
    # plt.ylim([0.0, 1.0])

    plt.xlabel('depth')
    plt.ylabel('entropy (bits)')

    plt.title("Entropy of CNN layers by depth")
    plt.legend()
    plt.tight_layout()
    plt.show()