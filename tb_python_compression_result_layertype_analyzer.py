import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, 'logs', 'compression_test_result_int8_64B.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()

filepath = comp_args.filepath
layer_types = ['All', 'Conv2d', 'ReLU', 'BatchNorm2d',]

yw, xw = 2, len(layer_types) // 2
fig, axes = plt.subplots(yw, xw, gridspec_kw={'width_ratios': [1] * xw})

for axis, layer_type in zip(axes.flatten(), layer_types):
    categories = []
    results = {}

    with open(filepath, 'rt') as file:
        raw_content = list(map(lambda x: x.split(','), file.readlines()))
        header = raw_content[0]
        content = sorted(raw_content[1:], key=lambda x: x[0])

        algo_names = list(map(lambda x: x.strip()[11:-1], header[3:]))

        for name in algo_names:
            results[name] = []
        results['total size'] = []

        for model_name, param_name, file_size, *comp_ratios in content:
            model_name = model_name.split('_')[0]
            curr_layer_type = param_name.split('_')[0]

            if layer_type != 'All' and curr_layer_type != layer_type:
                continue

            if model_name not in categories:
                categories.append(model_name)
                for nidx, name in enumerate(algo_names):
                    results[name].append(int(file_size) / float(comp_ratios[nidx]))
                results['total size'].append(int(file_size))
            else:
                for nidx, name in enumerate(algo_names):
                    results[name][categories.index(model_name)] += int(file_size) / float(comp_ratios[nidx])
                results['total size'][categories.index(model_name)] += int(file_size)

        for name in algo_names:
            results[name] = np.array(results['total size']) / np.array(results[name])
        del results['total size']

    width_max = 0.8
    width = width_max / len(results.keys())

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
        axis.bar(xval, val, width=width, label=key)
        for i, j in zip(xval, val):
            axis.annotate(f"{j:.2f}", xy=(i, j+0.06), ha='center', size=5)
    axis.set_xticks(x_axis, categories, rotation=0, ha='center')
    axis.set_title(f"layer type: {layer_type}")
    axis.set_ylim([0.0, 4])

plt.suptitle("Compression Algorithm Test on DL activations by layer types (INT8)")
plt.legend()
plt.tight_layout()
plt.show()